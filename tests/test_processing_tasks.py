from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from helpers import png
from image_api.app import create_app
from image_api.config import Settings
from image_api.lane import GpuLane
from image_api.processing import ProcessingRunner, recover_processing_tasks
from image_api.store import TaskStore
from image_api.workers import FakeWorkerClient


def setup(
    tmp_path: Path, **overrides: object
) -> tuple[TestClient, TaskStore, FakeWorkerClient, Settings]:
    settings = Settings.for_tests(tmp_path, **overrides)
    store = TaskStore(settings.database_path, settings.max_queue_depth)
    workers = FakeWorkerClient()
    client = TestClient(create_app(settings=settings, store=store, workers=workers))
    return client, store, workers, settings


def admit_upscale(
    client: TestClient,
    *,
    key: str = "upscale-key-0001",
    source: bytes | None = None,
    outscale: float = 2,
    tile: int = 512,
):
    return client.post(
        f"/v1/upscale-tasks?model=RealESRGAN_x4plus&outscale={outscale}&tile={tile}",
        files={"file": ("source.png", source or png("RGB", (1024, 1024)), "image/png")},
        headers={"Idempotency-Key": key},
    )


def admit_background(
    client: TestClient,
    *,
    key: str = "background-key-01",
    source: bytes | None = None,
    suffix: str = "",
):
    return client.post(
        "/v1/background-removal-tasks"
        "?model=birefnet-hr-matting&alpha_blur=1.5&alpha_erode=2&alpha_dilate=3"
        "&alpha_threshold=4&birefnet_inference_size=4096"
        "&birefnet_foreground_refinement=true&model_input_size=1024"
        "&despill_enabled=true&despill_color=custom&despill_hex_color=12Ab34"
        f"{suffix}",
        files={"file": ("source.png", source or png("RGBA", (8192, 8192)), "image/png")},
        headers={"Idempotency-Key": key},
    )


def test_admission_is_durable_and_returns_before_model_invocation(tmp_path: Path) -> None:
    client, store, workers, settings = setup(
        tmp_path,
        processing_max_input_pixels=1024 * 1024,
        processing_max_decoded_input_bytes=1024 * 1024 * 4,
        processing_max_output_pixels=2048 * 2048,
        processing_max_decoded_output_bytes=2048 * 2048 * 4,
    )
    source = png("RGB", (1024, 1024))

    response = admit_upscale(client, source=source)

    assert response.status_code == 202
    body = response.json()
    assert body == {
        "taskId": body["taskId"],
        "status": "queued",
        "capability": "upscale",
        "model": "RealESRGAN_x4plus",
        "sourceSha256": hashlib.sha256(source).hexdigest(),
        "requestedWidth": 1024,
        "requestedHeight": 1024,
        "expectedWidth": 2048,
        "expectedHeight": 2048,
        "expectedMode": "RGB",
    }
    assert workers.model_invocations == 0
    task = store.get(body["taskId"])
    assert task.task_kind == "upscale"
    assert task.request["outscale"] == 2.0
    assert task.request["tile"] == 512
    assert (settings.source_dir / task.request["source_image_name"]).read_bytes() == source


def test_exact_replay_returns_same_task_and_conflict_is_409(tmp_path: Path) -> None:
    client, store, workers, _ = setup(tmp_path)
    source = png("RGB", (8, 6))

    first = admit_upscale(client, source=source)
    replay = admit_upscale(client, source=source)
    conflict = admit_upscale(client, source=source, tile=256)

    assert first.status_code == replay.status_code == 202
    assert replay.json() == first.json()
    assert conflict.status_code == 409
    assert store.count() == 1
    assert workers.model_invocations == 0


def test_workers_claim_only_their_exact_capability(tmp_path: Path) -> None:
    client, store, _, _ = setup(tmp_path)
    generation = store.admit(
        "generation-key",
        {
            "width": 256,
            "height": 256,
            "seed": 1,
            "sampler_preset": "V4_TURBO_12",
            "structured_caption": {"description": "bee"},
        },
    )
    upscale = admit_upscale(client, source=png("RGB", (8, 6))).json()["taskId"]
    background = admit_background(
        client, source=png("RGBA", (8, 6)), key="background-key-02"
    ).json()["taskId"]

    assert store.claim_next("generation-worker", "generation").task_id == generation.task_id
    assert store.claim_next("generation-worker", "generation") is None
    assert store.claim_next("upscale-worker", "upscale").task_id == upscale
    assert store.claim_next("upscale-worker", "upscale") is None
    assert store.claim_next("background-worker", "background-removal").task_id == background
    assert store.claim_next("background-worker", "background-removal") is None


def test_runner_publishes_exact_output_metadata_and_lost_response_replay_is_single_call(
    tmp_path: Path,
) -> None:
    client, store, _, settings = setup(tmp_path)
    admitted = admit_upscale(client, source=png("RGBA", (8, 6)))
    task_id = admitted.json()["taskId"]
    calls = 0

    def model(_source: Path, request: dict[str, object]) -> bytes:
        nonlocal calls
        calls += 1
        return png("RGB", (request["expected_width"], request["expected_height"]))

    runner = ProcessingRunner(
        "upscale",
        store,
        GpuLane(settings.gpu_lane_path),
        settings.source_dir,
        settings.output_dir,
        model,
        settings,
    )
    assert runner.run_one() is True
    assert runner.run_one() is False

    replay = admit_upscale(client, source=png("RGBA", (8, 6)))
    status = client.get(f"/v1/upscale-tasks/{task_id}")
    download = client.get(f"/v1/upscale-tasks/{task_id}/image")
    record = store.get(task_id)

    assert replay.json()["taskId"] == task_id
    assert replay.json()["status"] == "succeeded"
    assert calls == 1
    assert status.status_code == download.status_code == 200
    assert download.content == png("RGB", (16, 12))
    assert record.output_sha256 == hashlib.sha256(download.content).hexdigest()
    assert (record.output_width, record.output_height, record.output_mode) == (16, 12, "RGB")
    assert status.json()["output"] == {
        "fileName": f"{task_id}.png",
        "sha256": record.output_sha256,
        "width": 16,
        "height": 12,
        "mode": "RGB",
    }


def test_status_and_result_behavior_for_all_states_and_corrupt_success(
    tmp_path: Path, caplog
) -> None:
    client, store, _, settings = setup(tmp_path)
    queued = admit_upscale(client, key="upscale-state-01", source=png("RGB", (8, 6))).json()[
        "taskId"
    ]
    running = admit_upscale(client, key="upscale-state-02", source=png("RGB", (8, 6))).json()[
        "taskId"
    ]
    assert store.claim_next("upscale", "upscale").task_id == queued
    assert store.claim_next("upscale", "upscale").task_id == running
    store.fail(running, "bounded_failure")

    assert client.get(f"/v1/upscale-tasks/{queued}").json()["status"] == "running"
    assert client.get(f"/v1/upscale-tasks/{queued}/image").status_code == 409
    failed = client.get(f"/v1/upscale-tasks/{running}").json()
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "bounded_failure"

    output = png("RGB", (16, 12))
    path = settings.output_dir / f"{queued}.png"
    path.write_bytes(output)
    store.succeed(
        queued,
        path.name,
        output_sha256=hashlib.sha256(output).hexdigest(),
        output_width=16,
        output_height=12,
        output_mode="RGB",
    )
    path.write_bytes(b"corrupt")
    with caplog.at_level("ERROR"):
        corrupt = client.get(f"/v1/upscale-tasks/{queued}/image")
    assert corrupt.status_code == 503
    assert "processing result validation failed" in caplog.text


def test_restart_reconciles_persisted_success_or_fails_without_rerun(tmp_path: Path) -> None:
    client, store, _, settings = setup(tmp_path)
    succeeded = admit_upscale(client, key="upscale-recover-1", source=png("RGB", (8, 6))).json()[
        "taskId"
    ]
    interrupted = admit_upscale(client, key="upscale-recover-2", source=png("RGB", (8, 6))).json()[
        "taskId"
    ]
    assert store.claim_next("dead", "upscale").task_id == succeeded
    assert store.claim_next("dead", "upscale").task_id == interrupted
    settings.output_dir.mkdir(exist_ok=True)
    (settings.output_dir / f"{succeeded}.png").write_bytes(png("RGB", (16, 12)))
    calls = 0

    def model(_source: Path, _request: dict[str, object]) -> bytes:
        nonlocal calls
        calls += 1
        return b"must not run"

    recovered = recover_processing_tasks(
        "upscale", store, settings.output_dir, settings.source_dir, settings
    )
    runner = ProcessingRunner(
        "upscale",
        store,
        GpuLane(settings.gpu_lane_path),
        settings.source_dir,
        settings.output_dir,
        model,
        settings,
    )

    assert recovered == 2
    assert store.get(succeeded).status == "succeeded"
    assert store.get(interrupted).status == "failed"
    assert store.get(interrupted).error_code == "worker_interrupted"
    assert runner.run_one() is False
    assert calls == 0


def test_background_8k_contract_binds_every_option_and_produces_rgba(tmp_path: Path) -> None:
    client, store, _, settings = setup(
        tmp_path,
        processing_max_upload_bytes=300_000_000,
        processing_max_request_bytes=305_000_000,
        processing_max_input_pixels=8192 * 8192,
        processing_max_output_pixels=8192 * 8192,
        processing_max_decoded_input_bytes=8192 * 8192 * 4,
        processing_max_decoded_output_bytes=8192 * 8192 * 4,
    )
    source = png("RGB", (64, 64))
    admitted = admit_background(client, source=source)
    assert admitted.status_code == 202
    task = store.get(admitted.json()["taskId"])
    expected_request = {
        "model": "birefnet-hr-matting",
        "alpha_blur": 1.5,
        "alpha_erode": 2,
        "alpha_dilate": 3,
        "alpha_threshold": 4,
        "birefnet_inference_size": 4096,
        "birefnet_foreground_refinement": True,
        "model_input_size": 1024,
        "despill_enabled": True,
        "despill_color": "custom",
        "despill_hex_color": "12ab34",
        "requested_width": 64,
        "requested_height": 64,
        "expected_width": 64,
        "expected_height": 64,
        "expected_mode": "RGBA",
    }
    assert {key: task.request[key] for key in expected_request} == expected_request

    def model(_source: Path, request: dict[str, object]) -> bytes:
        return png("RGBA", (request["expected_width"], request["expected_height"]))

    runner = ProcessingRunner(
        "background-removal",
        store,
        GpuLane(settings.gpu_lane_path),
        settings.source_dir,
        settings.output_dir,
        model,
        settings,
    )
    assert runner.run_one() is True
    result = client.get(f"/v1/background-removal-tasks/{task.task_id}/image")
    assert result.status_code == 200
    with Image.open(BytesIO(result.content)) as image:
        assert image.size == (64, 64)
        assert image.mode == "RGBA"


def test_processing_limits_are_route_scoped_and_sync_routes_remain_compatible(
    tmp_path: Path,
) -> None:
    client, _, workers, settings = setup(
        tmp_path,
        max_request_bytes=100,
        max_upload_bytes=90,
        processing_max_request_bytes=10_000,
        processing_max_upload_bytes=9_000,
    )
    source = png("RGB", (8, 6))

    task = admit_upscale(client, source=source)
    sync = client.post(
        "/v1/upscale?model=RealESRGAN_x4plus&outscale=2&tile=512",
        files={"file": ("source.png", source, "image/png")},
    )
    generation = client.post(
        "/v1/generations",
        json={
            "width": 256,
            "height": 256,
            "seed": 1,
            "sampler_preset": "V4_TURBO_12",
            "structured_caption": {"description": "x" * 200},
        },
        headers={"Idempotency-Key": "generation-limit-1"},
    )

    assert task.status_code == 202
    assert sync.status_code == 200
    assert workers.last_upscale == {"model": "RealESRGAN_x4plus", "outscale": 2.0, "tile": 512}
    assert generation.status_code == 413
    assert settings.processing_max_request_bytes > settings.max_request_bytes
