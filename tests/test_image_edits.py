from __future__ import annotations

import hashlib
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from helpers import png
from image_api import generation
from image_api.app import create_app
from image_api.config import Settings
from image_api.generation import GenerationRunner, reconcile_source_files, recover_interrupted_tasks
from image_api.lane import GpuLane
from image_api.store import TaskStore
from image_api.workers import FakeWorkerClient

OFFICIAL_EDIT_SIZE = (1408, 752)


def _client(tmp_path):
    settings = Settings.for_tests(tmp_path)
    store = TaskStore(settings.database_path)
    client = TestClient(create_app(settings=settings, store=store, workers=FakeWorkerClient()))
    return client, settings, store


def _edit(client: TestClient, key: str, image: bytes | None = None, **fields: object):
    form = {
        "model": "longcat-image-edit",
        "prompt": "make the bee blue",
        "seed": "43",
        "negative_prompt": "",
    }
    form.update({name: str(value) for name, value in fields.items()})
    return client.post(
        "/v1/image-edits",
        data=form,
        files={"file": ("source.png", image or png("RGBA", (13, 7)), "image/png")},
        headers={"Idempotency-Key": key},
    )


def test_image_edit_is_normalized_and_durable_before_202(tmp_path) -> None:
    client, settings, store = _client(tmp_path)
    source = png("RGBA", (13, 7))

    response = _edit(client, "edit-durable", source)

    assert response.status_code == 202
    task = store.get(response.json()["taskId"])
    source_path = settings.source_dir / task.request["source_image_name"]
    assert source_path.is_file()
    assert task.request["source_image_sha256"] == hashlib.sha256(source).hexdigest()
    with Image.open(source_path) as image:
        assert image.mode == "RGB"
        assert image.size == (13, 7)
    assert "source_image" not in response.text


def test_image_edit_replay_and_conflict_cover_image_and_every_parameter(tmp_path) -> None:
    client, settings, store = _client(tmp_path)
    first = _edit(client, "edit-replay")
    replay = _edit(client, "edit-replay")
    assert first.status_code == replay.status_code == 202
    assert first.json() == replay.json()

    conflicts = [
        _edit(client, "edit-replay", png("RGB", (14, 7))),
        _edit(client, "edit-replay", prompt="different"),
        _edit(client, "edit-replay", seed=44),
        _edit(client, "edit-replay", negative_prompt="text"),
        _edit(client, "edit-replay", model="longcat-image-edit-turbo"),
    ]
    assert [response.status_code for response in conflicts] == [409] * 5
    assert store.count() == 1
    assert len(list(settings.source_dir.glob("*.png"))) == 1


def test_image_edit_replay_survives_gateway_restart_without_duplicate_task(tmp_path) -> None:
    settings = Settings.for_tests(tmp_path)
    database = TaskStore(settings.database_path)
    first_client = TestClient(
        create_app(settings=settings, store=database, workers=FakeWorkerClient())
    )
    first = _edit(first_client, "restart-replay")
    assert first.status_code == 202

    restarted_client = TestClient(
        create_app(
            settings=settings,
            store=TaskStore(settings.database_path),
            workers=FakeWorkerClient(),
        )
    )
    replay = _edit(restarted_client, "restart-replay")
    assert replay.status_code == 202
    assert replay.json()["taskId"] == first.json()["taskId"]
    assert TaskStore(settings.database_path).count() == 1
    assert len(list(settings.source_dir.glob("*.png"))) == 1


def test_image_edit_validates_source_and_fields_before_admission(tmp_path) -> None:
    client, _, store = _client(tmp_path)
    invalid = _edit(client, "invalid-image", b"not an image")
    missing_prompt = _edit(client, "invalid-prompt", prompt="")
    unsupported = _edit(client, "invalid-model", model="ideogram-4-nf4")
    assert [invalid.status_code, missing_prompt.status_code, unsupported.status_code] == [
        400,
        422,
        422,
    ]
    assert store.count() == 0


def test_image_edit_keeps_established_body_limit_while_processing_uses_8k_limit(
    tmp_path,
) -> None:
    settings = Settings.for_tests(
        tmp_path,
        max_request_bytes=21_000_000,
        max_upload_bytes=20_000_000,
        processing_max_request_bytes=285_000_000,
        processing_max_upload_bytes=280_000_000,
    )
    store = TaskStore(settings.database_path)
    worker = FakeWorkerClient()
    client = TestClient(create_app(settings=settings, store=store, workers=worker))

    rejected_edit = client.post(
        "/v1/image-edits",
        data={
            "model": "longcat-image-edit",
            "prompt": "keep the established boundary",
            "seed": "43",
        },
        files={"file": ("source.png", png("RGB", (2, 2)), "image/png")},
        headers={
            "Idempotency-Key": "edit-size-boundary",
            "Content-Length": str(settings.max_request_bytes + 1),
        },
    )
    admitted_processing = client.post(
        "/v1/upscale?model=RealESRGAN_x4plus&outscale=2&tile=512",
        files={"file": ("source.png", png("RGB", (2, 2)), "image/png")},
        headers={"Content-Length": str(settings.processing_max_request_bytes)},
    )

    assert rejected_edit.status_code == 413
    assert store.count() == 0
    assert admitted_processing.status_code == 200
    assert worker.model_invocations == 1


def test_edit_runner_consumes_persisted_image_and_atomically_publishes_rgb_png(tmp_path) -> None:
    client, settings, store = _client(tmp_path)
    admitted = _edit(client, "edit-run", model="longcat-image-edit-turbo")
    task_id = admitted.json()["taskId"]
    source_path = settings.source_dir / store.get(task_id).request["source_image_name"]
    calls: list[tuple[dict[str, object], bytes]] = []

    def model(request: dict[str, object]) -> bytes:
        source = (settings.source_dir / str(request["source_image_name"])).read_bytes()
        calls.append((request, source))
        return png("RGB", OFFICIAL_EDIT_SIZE)

    runner = GenerationRunner(
        store,
        GpuLane(settings.gpu_lane_path),
        settings.output_dir,
        model,
        source_dir=settings.source_dir,
    )
    assert runner.run_one() is True
    complete = store.get(task_id)
    assert complete.status == "succeeded"
    assert calls[0][0]["model"] == "longcat-image-edit-turbo"
    with Image.open(BytesIO(calls[0][1])) as source:
        assert source.mode == "RGB"
        assert source.size == (13, 7)
    result = client.get(f"/v1/image-edits/{task_id}/image")
    assert result.status_code == 200
    with Image.open(BytesIO(result.content)) as image:
        assert image.mode == "RGB"
        assert image.size == OFFICIAL_EDIT_SIZE
    assert not list(settings.output_dir.glob("*.tmp"))
    assert not source_path.exists()


def test_edit_runner_rejects_wrong_official_output_dimensions(tmp_path) -> None:
    client, settings, store = _client(tmp_path)
    task_id = _edit(client, "edit-wrong-size").json()["taskId"]
    runner = GenerationRunner(
        store,
        GpuLane(settings.gpu_lane_path),
        settings.output_dir,
        lambda _request: png("RGB", (1, 1)),
        source_dir=settings.source_dir,
    )

    assert runner.run_one() is True
    failed = store.get(task_id)
    assert failed.status == "failed"
    assert failed.error_code == "worker_interrupted"
    assert not (settings.output_dir / f"{task_id}.png").exists()
    assert not list(settings.source_dir.glob("*.png"))


def test_edit_restart_reconciles_valid_result_without_readmission(tmp_path) -> None:
    client, settings, store = _client(tmp_path)
    admitted = _edit(client, "edit-restart")
    task_id = admitted.json()["taskId"]
    assert store.claim_next("crashed").task_id == task_id
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / f"{task_id}.png").write_bytes(png("RGB", OFFICIAL_EDIT_SIZE))

    assert recover_interrupted_tasks(store, settings.output_dir, settings.source_dir) == 1
    assert store.get(task_id).status == "succeeded"
    assert not list(settings.source_dir.glob("*.png"))
    replay = _edit(client, "edit-restart")
    assert replay.status_code == 202
    assert replay.json()["taskId"] == task_id
    assert store.count() == 1


def test_edit_restart_rejects_wrong_dimensions_and_cleans_source(tmp_path) -> None:
    client, settings, store = _client(tmp_path)
    task_id = _edit(client, "edit-restart-wrong-size").json()["taskId"]
    claimed = store.claim_next("crashed")
    assert claimed is not None
    assert claimed.task_id == task_id
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / f"{task_id}.png").write_bytes(png("RGB", (1, 1)))

    assert recover_interrupted_tasks(store, settings.output_dir, settings.source_dir) == 1
    assert store.get(task_id).status == "failed"
    assert not list(settings.source_dir.glob("*.png"))


@pytest.mark.parametrize(
    ("field", "value"),
    [("source_width", 0), ("source_width", True), ("source_height", -1), ("source_height", 7.0)],
)
def test_edit_runner_rejects_invalid_persisted_source_dimensions(
    tmp_path, field: str, value: object
) -> None:
    settings = Settings.for_tests(tmp_path)
    store = TaskStore(settings.database_path)
    request: dict[str, object] = {
        "task_type": "image-edit",
        "source_width": 13,
        "source_height": 7,
    }
    request[field] = value
    task = store.admit(f"invalid-{field}-{value!r}", request)

    runner = GenerationRunner(
        store,
        GpuLane(settings.gpu_lane_path),
        settings.output_dir,
        lambda _request: png("RGB", OFFICIAL_EDIT_SIZE),
        source_dir=settings.source_dir,
    )
    assert runner.run_one() is True
    assert store.get(task.task_id).status == "failed"


def test_interrupted_edit_without_output_cleans_source(tmp_path) -> None:
    client, settings, store = _client(tmp_path)
    task_id = _edit(client, "edit-interrupted").json()["taskId"]
    claimed = store.claim_next("crashed")
    assert claimed is not None
    assert claimed.task_id == task_id

    assert recover_interrupted_tasks(store, settings.output_dir, settings.source_dir) == 1
    assert store.get(task_id).status == "failed"
    assert not list(settings.source_dir.glob("*.png"))


def test_startup_source_reconciliation_removes_only_owned_orphans(tmp_path) -> None:
    settings = Settings.for_tests(tmp_path)
    store = TaskStore(settings.database_path)
    settings.source_dir.mkdir(parents=True)
    canonical = f"{'a' * 64}-{'b' * 64}.png"
    active = f"{'d' * 64}-{'e' * 64}.png"
    temporary = f".{canonical}.{'c' * 32}.tmp"
    (settings.source_dir / canonical).write_bytes(b"orphan")
    (settings.source_dir / active).write_bytes(b"active")
    (settings.source_dir / temporary).write_bytes(b"partial")
    (settings.source_dir / "unrelated.png").write_bytes(b"keep")
    store.admit("active-source", {"source_image_name": active})

    assert reconcile_source_files(store, settings.source_dir) == 2
    assert {path.name for path in settings.source_dir.iterdir()} == {
        ".source-files.lock",
        active,
        "unrelated.png",
    }


def test_shared_source_is_removed_only_after_last_active_task_terminates(tmp_path) -> None:
    client, settings, store = _client(tmp_path)
    source = png("RGB", (13, 7))
    first = _edit(client, "shared-first", source).json()["taskId"]
    second = _edit(client, "shared-second", source).json()["taskId"]
    source_path = settings.source_dir / store.get(first).request["source_image_name"]
    runner = GenerationRunner(
        store,
        GpuLane(settings.gpu_lane_path),
        settings.output_dir,
        lambda _request: png("RGB", OFFICIAL_EDIT_SIZE),
        source_dir=settings.source_dir,
    )

    assert runner.run_one() is True
    assert store.get(first).status == "succeeded"
    assert store.get(second).status == "queued"
    assert source_path.exists()

    assert runner.run_one() is True
    assert store.get(second).status == "succeeded"
    assert not source_path.exists()


def test_source_cleanup_failure_does_not_corrupt_terminal_task_state(tmp_path, monkeypatch) -> None:
    client, settings, store = _client(tmp_path)
    task_id = _edit(client, "cleanup-failure").json()["taskId"]
    source_path = settings.source_dir / store.get(task_id).request["source_image_name"]
    real_unlink = generation.os.unlink

    def fail_source_unlink(path: str | bytes) -> None:
        if generation.os.fspath(path) == generation.os.fspath(source_path):
            raise OSError("simulated source cleanup failure")
        real_unlink(path)

    monkeypatch.setattr(generation.os, "unlink", fail_source_unlink)
    runner = GenerationRunner(
        store,
        GpuLane(settings.gpu_lane_path),
        settings.output_dir,
        lambda _request: png("RGB", OFFICIAL_EDIT_SIZE),
        source_dir=settings.source_dir,
    )

    assert runner.run_one() is True
    assert store.get(task_id).status == "succeeded"
    assert source_path.exists()
