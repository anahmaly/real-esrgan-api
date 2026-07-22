from __future__ import annotations

import asyncio
import io
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from helpers import png
from image_api.config import Settings
from image_api.app import create_app
from image_api.store import TaskStore
from image_api.workers import FakeWorkerClient
from image_api.images import (
    ImageInfo,
    ImageTooLarge,
    InvalidWorkerImage,
    processing_output_size,
    validate_image,
    validate_png_output,
)
from image_api.workers import HttpWorkerClient
from image_api_workers import background as background_worker
from image_api_workers import upscale
from image_api_workers.uploads import read_bounded_upload

SQUARE_4K = 4096
SQUARE_8K = 8192
SQUARE_8K_PIXELS = 67_108_864


def test_8k_defaults_are_processing_only_and_cover_rgba_contract(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IMAGE_API_STATE_DIR", str(tmp_path))
    for name in (
        "IMAGE_API_MAX_REQUEST_BYTES",
        "IMAGE_API_MAX_UPLOAD_BYTES",
        "IMAGE_API_MAX_INPUT_WIDTH",
        "IMAGE_API_MAX_INPUT_HEIGHT",
        "IMAGE_API_MAX_INPUT_PIXELS",
        "IMAGE_API_MAX_OUTPUT_PIXELS",
        "IMAGE_API_MAX_DECODED_INPUT_BYTES",
        "IMAGE_API_MAX_DECODED_OUTPUT_BYTES",
        "IMAGE_API_PROCESSING_MAX_REQUEST_BYTES",
        "IMAGE_API_PROCESSING_MAX_UPLOAD_BYTES",
        "IMAGE_API_PROCESSING_MAX_ENCODED_OUTPUT_BYTES",
        "IMAGE_API_PROCESSING_MAX_INPUT_WIDTH",
        "IMAGE_API_PROCESSING_MAX_INPUT_HEIGHT",
        "IMAGE_API_PROCESSING_MAX_INPUT_PIXELS",
        "IMAGE_API_PROCESSING_MAX_OUTPUT_PIXELS",
        "IMAGE_API_PROCESSING_MAX_DECODED_INPUT_BYTES",
        "IMAGE_API_PROCESSING_MAX_DECODED_OUTPUT_BYTES",
        "IMAGE_API_PROCESSING_MAX_NATIVE_WIDTH",
        "IMAGE_API_PROCESSING_MAX_NATIVE_HEIGHT",
        "IMAGE_API_PROCESSING_MAX_NATIVE_PIXELS",
        "IMAGE_API_PROCESSING_MAX_NATIVE_BYTES",
        "IMAGE_API_WORKER_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.max_request_bytes == 21_000_000
    assert settings.max_upload_bytes == 20_000_000
    assert settings.max_input_width == 10_000
    assert settings.max_input_height == 10_000
    assert settings.max_input_pixels == 40_000_000
    assert settings.max_output_pixels == 80_000_000
    assert settings.processing_max_input_width == SQUARE_8K
    assert settings.processing_max_input_height == SQUARE_8K
    assert settings.processing_max_input_pixels == SQUARE_8K_PIXELS
    assert settings.processing_max_output_pixels == SQUARE_8K_PIXELS
    assert settings.processing_max_decoded_input_bytes >= SQUARE_8K_PIXELS * 4
    assert settings.processing_max_decoded_output_bytes >= SQUARE_8K_PIXELS * 4
    assert settings.admit_upscale_processing(SQUARE_4K, SQUARE_4K) == (16384, 16384)
    assert settings.processing_max_request_bytes > settings.processing_max_upload_bytes
    assert settings.processing_max_encoded_output_bytes > 40_000_000
    assert settings.worker_timeout_seconds >= 600


def test_8k_limits_and_timeout_accept_environment_overrides(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IMAGE_API_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("IMAGE_API_PROCESSING_MAX_ENCODED_OUTPUT_BYTES", "345000000")
    monkeypatch.setenv("IMAGE_API_PROCESSING_MAX_DECODED_INPUT_BYTES", "300000000")
    monkeypatch.setenv("IMAGE_API_PROCESSING_MAX_DECODED_OUTPUT_BYTES", "310000000")
    monkeypatch.setenv("IMAGE_API_PROCESSING_MAX_NATIVE_BYTES", "4000000000")
    monkeypatch.setenv("IMAGE_API_WORKER_TIMEOUT_SECONDS", "1234")

    settings = Settings.from_env()

    assert settings.processing_max_encoded_output_bytes == 345_000_000
    assert settings.processing_max_decoded_input_bytes == 300_000_000
    assert settings.processing_max_decoded_output_bytes == 310_000_000
    assert settings.processing_max_native_bytes == 4_000_000_000
    assert settings.worker_timeout_seconds == 1234

    monkeypatch.setenv("IMAGE_API_WORKER_TIMEOUT_SECONDS", "nan")
    with pytest.raises(ValueError, match="finite"):
        Settings.from_env()


def test_exact_staged_square_dimension_math_without_allocating_8k_images() -> None:
    assert processing_output_size(ImageInfo(1024, 1024, "RGB"), 4) == (4096, 4096)
    assert processing_output_size(ImageInfo(4096, 4096, "RGB"), 2) == (8192, 8192)


def test_8k_output_admission_is_exact_and_too_large_is_rejected() -> None:
    settings = Settings.for_tests(
        Path("/tmp/image-api-boundary-test"),
        processing_max_input_width=SQUARE_8K,
        processing_max_input_height=SQUARE_8K,
        processing_max_input_pixels=SQUARE_8K_PIXELS,
        processing_max_output_pixels=SQUARE_8K_PIXELS,
        processing_max_decoded_input_bytes=SQUARE_8K_PIXELS * 4,
        processing_max_decoded_output_bytes=SQUARE_8K_PIXELS * 4,
    )
    assert settings.admit_processing_output_dimensions(SQUARE_8K, SQUARE_8K) == (
        SQUARE_8K,
        SQUARE_8K,
    )
    with pytest.raises(ImageTooLarge):
        settings.admit_processing_output_dimensions(SQUARE_8K + 1, SQUARE_8K)

    constrained = Settings.for_tests(
        Path("/tmp/image-api-processing-boundary-test"),
        processing_max_native_width=16383,
    )
    with pytest.raises(ImageTooLarge, match="native processing"):
        constrained.admit_upscale_processing(SQUARE_4K, SQUARE_4K)


def test_decoded_memory_is_bounded_independently_of_encoded_bytes() -> None:
    with pytest.raises(ImageTooLarge, match="decoded"):
        validate_image(
            png("RGBA", (8, 8)),
            max_bytes=1_000_000,
            max_width=100,
            max_height=100,
            max_pixels=10_000,
            max_decoded_bytes=255,
        )


def test_encoded_worker_output_ceiling_is_independent() -> None:
    with pytest.raises(ImageTooLarge, match="encoded"):
        validate_png_output(
            png("RGBA", (8, 8)),
            expected_size=(8, 8),
            required_mode="RGBA",
            max_bytes=10,
            max_pixels=64,
            max_decoded_bytes=256,
        )


def test_processing_worker_upload_ceiling_is_bounded_and_closes_input() -> None:
    accepted = UploadFile(io.BytesIO(b"1234"), filename="input.png")
    assert asyncio.run(read_bounded_upload(accepted, 4)) == b"1234"
    assert accepted.file.closed

    rejected = UploadFile(io.BytesIO(b"12345"), filename="input.png")
    with pytest.raises(HTTPException) as failure:
        asyncio.run(read_bounded_upload(rejected, 4))
    assert failure.value.status_code == 413
    assert rejected.file.closed


def test_http_worker_timeout_and_streamed_output_spool_are_wired() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=png("RGB", (2, 2)))

    client = HttpWorkerClient(
        "http://upscale-worker",
        "http://background-worker",
        timeout_seconds=987,
        max_output_bytes=1_000_000,
        transport=httpx.MockTransport(handler),
    )

    result = client.upscale(io.BytesIO(png()), model="RealESRGAN_x4plus", outscale=2, tile=512)

    assert client.client.timeout.read == 987
    assert not isinstance(result, bytes)
    assert result.read(8) == b"\x89PNG\r\n\x1a\n"
    result.close()


def test_realesrgan_worker_discards_alpha_before_backend() -> None:
    image, size = upscale._load_rgb_source(png("RGBA", (3, 2)))

    assert image.mode == "RGB"
    assert size == (3, 2)


def test_realesrgan_large_input_cannot_disable_tiling() -> None:
    assert upscale._effective_tile(0, 1024, 1024) == 0
    assert upscale._effective_tile(0, 4096, 4096) == 512
    assert upscale._effective_tile(0, 8192, 8192) == 512
    assert upscale._effective_tile(768, 4096, 4096) == 768
    with pytest.raises(ValueError):
        upscale._effective_tile(31, 4096, 4096)


def test_processing_workers_reject_configured_pixel_overflow_without_large_fixture(
    monkeypatch,
) -> None:
    monkeypatch.setenv("IMAGE_API_PROCESSING_MAX_INPUT_PIXELS", "5")
    with pytest.raises(ImageTooLarge):
        upscale._load_rgb_source(png("RGB", (3, 2)))
    with pytest.raises(ImageTooLarge):
        background_worker._validate_worker_dimensions(3, 2, output=False)


def test_worker_output_requires_exact_dimensions_and_rgb_for_upscale() -> None:
    with pytest.raises(InvalidWorkerImage):
        validate_png_output(
            png("RGBA", (8, 8)),
            expected_size=(8, 8),
            required_mode="RGB",
            max_bytes=1_000_000,
            max_pixels=64,
            max_decoded_bytes=256,
        )


def test_openapi_keeps_birefnet_inference_bounded_at_4096(tmp_path: Path) -> None:
    settings = Settings.for_tests(tmp_path)
    client = TestClient(
        create_app(
            settings=settings,
            store=TaskStore(settings.database_path),
            workers=FakeWorkerClient(),
        )
    )

    operation = client.get("/openapi.json").json()["paths"]["/v1/background-removal"]["post"]
    parameter = next(
        item for item in operation["parameters"] if item["name"] == "birefnet_inference_size"
    )

    assert parameter["schema"]["maximum"] == 4096
    assert parameter["schema"]["default"] == 2048
