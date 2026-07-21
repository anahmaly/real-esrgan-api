from __future__ import annotations

import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def repository_text() -> str:
    parts = []
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or ".venv" in path.parts
            or "__pycache__" in path.parts
        ):
            continue
        if (
            "tests" in path.parts
            or path.name == "test_cutover_contract.py"
            or "licenses" in path.parts
            or path.name in {"NOTICE.md", "README.md"}
        ):
            continue
        if path.suffix in {".pyc", ".png"} or path.name == "uv.lock":
            continue
        try:
            parts.append(f"{path.relative_to(ROOT)}\n{path.read_text()}")
        except UnicodeDecodeError:
            pass
    return "\n".join(parts)


def test_old_public_identity_and_route_contract_are_absent() -> None:
    text = repository_text()
    assert "real-esrgan-api" not in text.lower()
    assert '"/upscale/"' not in text
    assert "REALESRGAN_MODEL" not in text
    assert "REALESRGAN_" not in text
    assert "/models/realesrgan" not in text
    compose = (ROOT / "compose.yml").read_text()
    assert "container_name: realesrgan" not in compose
    assert "\n  realesrgan:" not in compose
    assert "image-api" in compose


def test_only_gateway_publishes_ports() -> None:
    compose = (ROOT / "compose.yml").read_text().splitlines()
    port_lines = [line for line in compose if line.strip() == "ports:"]
    assert len(port_lines) == 1


def test_compose_has_no_legacy_background_model_mount_contract() -> None:
    compose = (ROOT / "compose.yml").read_text()
    assert "/models/rembg" not in compose
    assert "IMAGE_API_REMBG_MODELS_PATH" not in compose
    assert "IMAGE_API_REMBG_WEIGHTS_PATH" not in compose
    assert "IMAGE_API_REMBG_WEIGHTS_HOST_PATH" not in compose


def test_pinned_upstream_sources_and_no_weight_download_commands() -> None:
    text = repository_text()
    assert "dd7b6fd434cff2077ce6e9a0cab46fe254f26f1f" in text
    assert "990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2" in text
    dockerfiles = "\n".join(path.read_text() for path in ROOT.glob("Dockerfile*"))
    assert "wget" not in dockerfiles
    assert "curl" not in dockerfiles
    assert "hf auth" not in dockerfiles.lower()


def test_background_install_pins_birefnet_runtime_dependency() -> None:
    dockerfiles = list(ROOT.glob("Dockerfile*"))
    background = (ROOT / "Dockerfile.background").read_text().replace("\\\n", " ")
    install_line = next(
        line.removeprefix("RUN ")
        for line in background.splitlines()
        if line.startswith("RUN ") and "rembg-api.git" in line
    )
    tokens = shlex.split(install_line)

    einops_tokens = [token for token in tokens if token.lower().startswith("einops")]
    assert einops_tokens == ["einops==0.8.2"]
    assert (
        "git+https://github.com/anahmaly/rembg-api.git@dd7b6fd434cff2077ce6e9a0cab46fe254f26f1f"
    ) in tokens
    assert "--break-system-packages" not in background
    assert all(
        "einops" not in path.read_text().lower()
        for path in dockerfiles
        if path.name != "Dockerfile.background"
    )


def test_generation_install_handles_pep_668_and_keeps_ideogram_pinned() -> None:
    dockerfiles = list(ROOT.glob("Dockerfile*"))
    generation = (ROOT / "Dockerfile.generation").read_text().replace("\\\n", " ")
    install_line = next(
        line.removeprefix("RUN ")
        for line in generation.splitlines()
        if line.startswith("RUN ") and "ideogram4.git" in line
    )
    tokens = shlex.split(install_line)

    assert tokens[:4] == ["python", "-m", "pip", "install"]
    assert "--break-system-packages" in tokens
    assert (
        "git+https://github.com/ideogram-oss/ideogram4.git@990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2"
    ) in tokens
    assert all(
        "--break-system-packages" not in path.read_text()
        for path in dockerfiles
        if path.name != "Dockerfile.generation"
    )
