from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text()
COMPOSE = (ROOT / "compose.yml").read_text()
MAIN = (ROOT / "app" / "main.py").read_text()


def _step_six_run_instruction():
    marker = "# 6. Install Real-ESRGAN deps"
    start = DOCKERFILE.index(marker)
    end = DOCKERFILE.index("# 7. Copy API code", start)
    step_lines = DOCKERFILE[start:end].splitlines()
    run_starts = [
        index
        for index, line in enumerate(step_lines)
        if line.lstrip().startswith("RUN ")
    ]
    assert len(run_starts) == 1

    command_lines = []
    continued = False
    for line in step_lines[run_starts[0] :]:
        stripped = line.strip()
        if stripped == "# install headless OpenCV explicitly":
            assert continued
            continue
        if command_lines and not continued:
            break

        continued = stripped.endswith("\\")
        command_lines.append(stripped.removesuffix("\\").rstrip())

    assert command_lines[0].startswith("RUN ")
    return " ".join(command_lines)[len("RUN ") :]


def test_lmdb_build_toolchain_is_one_fail_closed_run_chain():
    expected_commands = [
        "apt-get update",
        "apt-get install -y --no-install-recommends build-essential",
        "sed -i '/torch/d' requirements.txt",
        "sed -i '/opencv-python/d' requirements.txt",
        "pip install --no-cache-dir basicsr==1.4.2 facexlib==0.2.5 gfpgan==1.3.8",
        "pip install --no-cache-dir -r requirements.txt",
        "pip install --no-cache-dir opencv-python-headless",
        "python setup.py develop",
        "apt-get purge -y --auto-remove build-essential",
        "rm -rf /var/lib/apt/lists/*",
    ]

    assert _step_six_run_instruction() == " && ".join(expected_commands)


def test_official_model_weight_urls_and_destinations_are_unchanged():
    assert (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
        "RealESRGAN_x4plus.pth"
    ) in DOCKERFILE
    assert "-O /Real-ESRGAN/weights/RealESRGAN_x4plus.pth" in DOCKERFILE
    assert (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/"
        "RealESRGAN_x4plus_anime_6B.pth"
    ) in DOCKERFILE
    assert (
        "-O /Real-ESRGAN/weights/RealESRGAN_x4plus_anime_6B.pth" in DOCKERFILE
    )


def test_gpu_runtime_and_upscale_selection_contracts_are_unchanged():
    assert "FROM python:3.8-slim-bullseye" in DOCKERFILE
    assert "torch==2.1.0+cu118" in DOCKERFILE
    assert "torchvision==0.16.0+cu118" in DOCKERFILE
    assert "torchaudio==2.1.0+cu118" in DOCKERFILE
    assert "--index-url https://download.pytorch.org/whl/cu118" in DOCKERFILE
    assert "EXPOSE 8000" in DOCKERFILE
    assert (
        'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]'
        in DOCKERFILE
    )

    assert '"8000:8000"' in COMPOSE
    assert "REALESRGAN_MODEL: ${REALESRGAN_MODEL:-RealESRGAN_x4plus}" in COMPOSE
    assert "capabilities: [gpu]" in COMPOSE
    assert "gpus: all" in COMPOSE

    assert "outscale: float = 2.0" in MAIN
    assert "tile: int = Query(" in MAIN
    assert 'os.getenv("REALESRGAN_MODEL")' in MAIN
