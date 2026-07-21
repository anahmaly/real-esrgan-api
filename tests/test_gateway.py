from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from helpers import png
from image_api.app import create_app
from image_api.config import Settings
from image_api.store import TaskStore
from image_api.workers import FakeWorkerClient


@pytest.fixture
def worker() -> FakeWorkerClient:
    return FakeWorkerClient()


@pytest.fixture
def client(tmp_path, worker: FakeWorkerClient) -> TestClient:
    settings = Settings.for_tests(tmp_path)
    return TestClient(
        create_app(settings=settings, store=TaskStore(settings.database_path), workers=worker)
    )


def test_health_and_models_do_not_load_models(client: TestClient, worker: FakeWorkerClient) -> None:
    health = client.get("/health")
    models = client.get("/v1/models")
    assert health.status_code == 200
    assert models.status_code == 200
    assert health.json()["service"] == "image-api"
    assert {item["capability"] for item in models.json()["models"]} == {
        "upscale",
        "background-removal",
        "generation",
        "image-editing",
    }
    assert [
        item["model"]
        for item in models.json()["models"]
        if item["capability"] == "background-removal"
    ] == ["bria-rmbg-2.0", "birefnet-hr-matting"]
    assert worker.model_invocations == 0
    assert worker.model_loads == 0
    assert "/models/" not in health.text + models.text


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_upscale_contract_and_exact_png_dimensions(
    client: TestClient, worker: FakeWorkerClient, mode: str
) -> None:
    response = client.post(
        "/v1/upscale?model=RealESRGAN_x4plus_anime_6B&outscale=2&tile=512",
        files={"file": ("input.png", png(mode, (9, 7)), "image/png")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    with Image.open(BytesIO(response.content)) as image:
        assert image.size == (18, 14)
        assert image.format == "PNG"
    assert worker.last_upscale == {
        "model": "RealESRGAN_x4plus_anime_6B",
        "outscale": 2.0,
        "tile": 512,
    }


def test_upscale_requires_explicit_model_and_rejects_old_route(client: TestClient) -> None:
    assert (
        client.post("/v1/upscale", files={"file": ("x.png", png(), "image/png")}).status_code == 422
    )
    assert (
        client.post("/upscale/", files={"file": ("x.png", png(), "image/png")}).status_code == 404
    )


@pytest.mark.parametrize(
    "query",
    [
        "model=RealESRGAN_x4plus&outscale=0.5&tile=512",
        "model=RealESRGAN_x4plus&outscale=2&tile=31",
        "model=unknown&outscale=2&tile=512",
    ],
)
def test_upscale_rejects_invalid_parameters_before_worker(
    client: TestClient, worker: FakeWorkerClient, query: str
) -> None:
    response = client.post(f"/v1/upscale?{query}", files={"file": ("x.png", png(), "image/png")})
    assert response.status_code == 422
    assert worker.model_invocations == 0


def test_invalid_and_oversized_images_reject_before_worker(
    tmp_path, worker: FakeWorkerClient
) -> None:
    settings = Settings.for_tests(tmp_path, max_upload_bytes=20, max_input_pixels=10)
    client = TestClient(
        create_app(settings=settings, store=TaskStore(settings.database_path), workers=worker)
    )
    too_many_pixels = png(size=(4, 3))
    assert (
        client.post(
            "/v1/upscale?model=RealESRGAN_x4plus&outscale=2&tile=512",
            files={"file": ("x.png", too_many_pixels, "image/png")},
        ).status_code
        == 413
    )
    assert (
        client.post(
            "/v1/upscale?model=RealESRGAN_x4plus&outscale=2&tile=512",
            files={"file": ("x.bin", b"bad", "application/octet-stream")},
        ).status_code
        == 400
    )
    assert worker.model_invocations == 0


def test_decompression_bomb_is_rejected_before_worker(
    tmp_path, worker: FakeWorkerClient, monkeypatch
) -> None:
    settings = Settings.for_tests(tmp_path)
    client = TestClient(
        create_app(settings=settings, store=TaskStore(settings.database_path), workers=worker)
    )
    monkeypatch.setattr("PIL.Image.MAX_IMAGE_PIXELS", 1)
    response = client.post(
        "/v1/upscale?model=RealESRGAN_x4plus&outscale=2&tile=512",
        files={"file": ("bomb.png", png(size=(8, 6)), "image/png")},
    )
    assert response.status_code == 413
    assert worker.model_invocations == 0


def test_declared_body_limit_rejects_before_route(tmp_path, worker: FakeWorkerClient) -> None:
    settings = Settings.for_tests(tmp_path, max_request_bytes=100, max_upload_bytes=90)
    client = TestClient(
        create_app(settings=settings, store=TaskStore(settings.database_path), workers=worker)
    )
    response = client.post(
        "/v1/upscale?model=RealESRGAN_x4plus&outscale=2&tile=512",
        files={"file": ("x.png", png(), "image/png")},
        headers={"Content-Length": "101"},
    )
    assert response.status_code == 413
    assert worker.model_invocations == 0
