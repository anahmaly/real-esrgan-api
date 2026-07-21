from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from image_api.app import create_app
from image_api.config import Settings
from image_api.store import TaskStore
from image_api.workers import FakeWorkerClient

ROOT = Path(__file__).resolve().parents[1]
STANDARD_REVISION = "7b54ef423aa7854be7861600024be5c56ab7875a"
TURBO_REVISION = "6a7262de5549f0bf0ec54c08ef7d283ef41f3214"


def test_models_health_and_openapi_distinguish_generation_from_editing(tmp_path) -> None:
    settings = Settings.for_tests(tmp_path)
    workers = FakeWorkerClient()
    workers.set_loaded("generation", "longcat-image-edit-turbo")
    client = TestClient(
        create_app(settings=settings, store=TaskStore(settings.database_path), workers=workers)
    )

    models = client.get("/v1/models").json()["models"]
    by_id = {entry["model"]: entry for entry in models}
    assert by_id["ideogram-4-nf4"]["capability"] == "generation"
    assert by_id["ideogram-4-nf4"]["acceptsSourceImage"] is False
    assert by_id["longcat-image-edit"]["defaults"] == {"guidanceScale": 4.5, "steps": 50}
    assert by_id["longcat-image-edit-turbo"]["defaults"] == {
        "guidanceScale": 1.0,
        "steps": 8,
    }
    health = client.get("/health").json()["capabilities"]
    assert health["generation"]["loaded"] is False
    assert health["image-editing"]["loadedModel"] == "longcat-image-edit-turbo"
    schema = client.get("/openapi.json").json()
    assert "/v1/image-edits" in schema["paths"]
    assert (
        "multipart/form-data"
        in schema["paths"]["/v1/image-edits"]["post"]["requestBody"]["content"]
    )
    assert "/v1/models/unload" in schema["paths"]


def test_compose_keeps_worker_controls_private_and_mounts_both_snapshots() -> None:
    compose = (ROOT / "compose.yml").read_text()
    assert compose.count("ports:") == 1
    assert 'expose:\n      - "9003"' in compose
    assert "IMAGE_API_GENERATION_WORKER_URL: http://generation-worker:9003" in compose
    assert "IMAGE_API_LONGCAT_EDIT_WEIGHTS_HOST_PATH" in compose
    assert "IMAGE_API_LONGCAT_EDIT_TURBO_WEIGHTS_HOST_PATH" in compose
    assert STANDARD_REVISION in compose
    assert TURBO_REVISION in compose
    test_compose = (ROOT / "compose.test.yml").read_text()
    assert "image_api_workers.fake_generation" in test_compose
    assert 'IMAGE_API_GENERATION_TEST_MODE: "true"' in test_compose


def test_generation_runtime_pins_and_import_checks_coexisting_pipelines() -> None:
    dockerfile = (ROOT / "Dockerfile.generation").read_text()
    for pin in (
        '"torch==2.11.0"',
        '"torchvision==0.26.0"',
        '"diffusers==0.37.0"',
        '"transformers==4.57.1"',
        '"accelerate==1.11.0"',
        '"safetensors==0.6.2"',
        "ideogram4.git@990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2",
    ):
        assert pin in dockerfile
    assert "from diffusers import LongCatImageEditPipeline" in dockerfile
    assert "import accelerate, diffusers, ideogram4, torch, transformers" in dockerfile
    workflow = (ROOT / ".github/workflows/image-api.yml").read_text()
    assert "docker build -f Dockerfile.generation" in workflow


def test_readme_documents_acquisition_limits_polling_and_unload() -> None:
    readme = (ROOT / "README.md").read_text()
    for required in (
        "hf download meituan-longcat/LongCat-Image-Edit",
        "hf download meituan-longcat/LongCat-Image-Edit-Turbo",
        STANDARD_REVISION,
        TURBO_REVISION,
        "approximately 29.3 GB",
        "18–19 GB VRAM",
        "RTX 4090",
        "approximately one megapixel",
        "no denoising/edit-strength parameter",
        "GET /v1/image-edits/{taskId}",
        "POST /v1/models/unload",
        "no LongCat quantization support",
    ):
        assert required.lower() in readme.lower()
