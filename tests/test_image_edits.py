from __future__ import annotations

import hashlib
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from helpers import png
from image_api.app import create_app
from image_api.config import Settings
from image_api.generation import GenerationRunner, recover_interrupted_tasks
from image_api.lane import GpuLane
from image_api.store import TaskStore
from image_api.workers import FakeWorkerClient


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


def test_edit_runner_consumes_persisted_image_and_atomically_publishes_rgb_png(tmp_path) -> None:
    client, settings, store = _client(tmp_path)
    admitted = _edit(client, "edit-run", model="longcat-image-edit-turbo")
    task_id = admitted.json()["taskId"]
    calls: list[tuple[dict[str, object], bytes]] = []

    def model(request: dict[str, object]) -> bytes:
        source = (settings.source_dir / str(request["source_image_name"])).read_bytes()
        calls.append((request, source))
        return png("RGB", (1024, 551))

    runner = GenerationRunner(store, GpuLane(settings.gpu_lane_path), settings.output_dir, model)
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
        assert image.size == (1024, 551)
    assert not list(settings.output_dir.glob("*.tmp"))


def test_edit_restart_reconciles_valid_result_without_readmission(tmp_path) -> None:
    client, settings, store = _client(tmp_path)
    admitted = _edit(client, "edit-restart")
    task_id = admitted.json()["taskId"]
    assert store.claim_next("crashed").task_id == task_id
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / f"{task_id}.png").write_bytes(png("RGB", (1024, 551)))

    assert recover_interrupted_tasks(store, settings.output_dir) == 1
    assert store.get(task_id).status == "succeeded"
    replay = _edit(client, "edit-restart")
    assert replay.status_code == 202
    assert replay.json()["taskId"] == task_id
    assert store.count() == 1
