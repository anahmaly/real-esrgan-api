from __future__ import annotations

import sys
from io import BytesIO
from types import ModuleType, SimpleNamespace
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from helpers import png
from image_api_workers import background

app = background.app


def _install_pr7_fakes(monkeypatch, calls: list[tuple[str, dict[str, object]]]) -> None:
    monkeypatch.setattr(background, "PeerEvictor", lambda _peers: lambda: None)
    package = ModuleType("rembg_api")
    package.__path__ = []  # type: ignore[attr-defined]

    birefnet = ModuleType("rembg_api.birefnet_hr")
    birefnet.DEFAULT_REVISION = "pinned"
    birefnet.BiRefNetConfig = lambda **kwargs: SimpleNamespace(**kwargs)

    def remove_with_birefnet(data: bytes, **kwargs: object) -> bytes:
        calls.append(("birefnet", kwargs))
        return png("RGBA", (13, 7))

    birefnet.remove_with_birefnet = remove_with_birefnet
    setattr(birefnet, "clear_cache", lambda: calls.append(("clear-birefnet", {})))

    bria = ModuleType("rembg_api.bria_rmbg")

    def remove_with_bria(data: bytes, **kwargs: object) -> bytes:
        calls.append(("bria", kwargs))
        return png("RGBA", (13, 7))

    bria.remove_with_bria_rmbg_2 = remove_with_bria
    setattr(bria, "clear_bria_backend_cache", lambda **kwargs: calls.append(("clear-bria", kwargs)))

    processing = ModuleType("rembg_api.image_processing")
    processing.AlphaOptions = lambda **kwargs: SimpleNamespace(**kwargs)
    processing.DespillOptions = lambda **kwargs: SimpleNamespace(**kwargs)
    processing.process_png_bytes = lambda data, **kwargs: data

    monkeypatch.setitem(sys.modules, "rembg_api", package)
    monkeypatch.setitem(sys.modules, "rembg_api.birefnet_hr", birefnet)
    monkeypatch.setitem(sys.modules, "rembg_api.bria_rmbg", bria)
    monkeypatch.setitem(sys.modules, "rembg_api.image_processing", processing)


@pytest.fixture(autouse=True)
def reset_active_background_model() -> Iterator[None]:
    background._active_model = None
    yield
    background._active_model = None


def _install_cuda_fake(monkeypatch) -> None:
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_health_is_ready_with_only_bria_and_birefnet_mounts(monkeypatch, tmp_path) -> None:
    bria = tmp_path / "bria"
    birefnet = tmp_path / "birefnet"
    bria.mkdir()
    birefnet.mkdir()
    monkeypatch.setenv("IMAGE_API_BRIA_WEIGHTS_PATH", str(bria))
    monkeypatch.setenv("IMAGE_API_BIREFNET_WEIGHTS_PATH", str(birefnet))
    monkeypatch.setenv("IMAGE_API_REMBG_WEIGHTS_PATH", str(tmp_path / "absent-legacy-models"))
    assert not (tmp_path / "absent-legacy-models").exists()
    _install_cuda_fake(monkeypatch)

    health = background._health()

    assert health["ready"] is True
    assert health["weightsAvailable"] is True


@pytest.mark.parametrize("missing", ["bria", "birefnet"])
def test_health_is_not_ready_when_a_remaining_mount_is_absent(
    monkeypatch, tmp_path, missing: str
) -> None:
    bria = tmp_path / "bria"
    birefnet = tmp_path / "birefnet"
    if missing != "bria":
        bria.mkdir()
    if missing != "birefnet":
        birefnet.mkdir()
    monkeypatch.setenv("IMAGE_API_BRIA_WEIGHTS_PATH", str(bria))
    monkeypatch.setenv("IMAGE_API_BIREFNET_WEIGHTS_PATH", str(birefnet))
    _install_cuda_fake(monkeypatch)

    health = background._health()

    assert health["ready"] is False
    assert health["weightsAvailable"] is False


@pytest.mark.parametrize(
    ("model", "expected", "query"),
    [
        (
            "bria-rmbg-2.0",
            "bria",
            "model_input_size=1536",
        ),
        (
            "birefnet-hr-matting",
            "birefnet",
            "birefnet_inference_size=3072&birefnet_foreground_refinement=true",
        ),
    ],
)
def test_pr7_backends_dispatch_with_bounded_options_and_rgba(
    monkeypatch, tmp_path, model: str, expected: str, query: str
) -> None:
    monkeypatch.setenv("IMAGE_API_STATE_DIR", str(tmp_path))
    calls: list[tuple[str, dict[str, object]]] = []
    _install_pr7_fakes(monkeypatch, calls)
    client = TestClient(app)
    response = client.post(
        f"/internal/background-removal?model={model}&{query}",
        files={"file": ("input.png", png("RGB", (13, 7)), "image/png")},
    )
    assert response.status_code == 200
    health = client.get("/health").json()
    assert health["loaded"] is True
    assert health["loadedModel"] == model
    assert calls[0][0] == expected
    if expected == "bria":
        assert calls[0][1]["model_input_size"] == 1536
    else:
        assert calls[0][1]["inference_size"] == 3072
        assert calls[0][1]["foreground_refinement"] is True
    with Image.open(BytesIO(response.content)) as image:
        assert image.format == "PNG"
        assert image.mode == "RGBA"
        assert image.size == (13, 7)


def test_switching_remaining_models_releases_resident_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("IMAGE_API_STATE_DIR", str(tmp_path))
    calls: list[tuple[str, dict[str, object]]] = []
    _install_pr7_fakes(monkeypatch, calls)
    client = TestClient(app)

    for model in ("bria-rmbg-2.0", "birefnet-hr-matting"):
        response = client.post(
            f"/internal/background-removal?model={model}",
            files={"file": ("input.png", png("RGB", (13, 7)), "image/png")},
        )
        assert response.status_code == 200

    assert [name for name, _ in calls] == [
        "bria",
        "clear-birefnet",
        "clear-bria",
        "birefnet",
    ]


def test_background_postprocessing_uses_configured_encoded_ceiling(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    _install_pr7_fakes(monkeypatch, calls)
    monkeypatch.setenv("IMAGE_API_PROCESSING_MAX_ENCODED_OUTPUT_BYTES", "345000000")
    observed: dict[str, object] = {}
    processing = sys.modules["rembg_api.image_processing"]

    def process(data: bytes, **kwargs: object) -> bytes:
        observed.update(kwargs)
        return data

    processing.process_png_bytes = process  # type: ignore[attr-defined]

    encoded = background._run_background(
        png("RGB", (13, 7)),
        model="birefnet-hr-matting",
        alpha_blur=0,
        alpha_erode=0,
        alpha_dilate=0,
        alpha_threshold=0,
        birefnet_inference_size=4096,
        birefnet_foreground_refinement=False,
        model_input_size=1024,
    )

    assert observed["max_encoded_bytes"] == 345_000_000
    assert encoded == png("RGBA", (13, 7))
