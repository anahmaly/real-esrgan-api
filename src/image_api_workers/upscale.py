from __future__ import annotations

import asyncio
import atexit
import gc
import io
import logging
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image

from image_api.images import validate_dimensions
from image_api.config import Settings
from image_api.lane import GpuLane
from image_api.processing import ProcessingRunner, recover_processing_tasks
from image_api.store import TaskStore
from image_api.workers import PeerEvictor
from image_api_workers.execution import execute_in_gpu_lane
from image_api_workers.uploads import read_bounded_upload

logger = logging.getLogger(__name__)
MODELS = {
    "RealESRGAN_x4plus": (23, "RealESRGAN_x4plus.pth"),
    "RealESRGAN_x4plus_anime_6B": (6, "RealESRGAN_x4plus_anime_6B.pth"),
}
_loaded_model_name: str | None = None
_model_lock = threading.RLock()


def _weights_dir() -> Path:
    return Path(os.getenv("IMAGE_API_UPSCALE_WEIGHTS_PATH", "/models/upscale"))


def _runtime_status() -> dict[str, object]:
    available = all((_weights_dir() / filename).is_file() for _, filename in MODELS.values())
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception as exc:
        logger.warning(
            "upscale CUDA runtime probe failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        cuda = False
    return {
        "ready": available and cuda,
        "loaded": _loaded_model_name is not None,
        "loadedModel": _loaded_model_name,
        "device": "cuda" if cuda else "unavailable",
        "weightsAvailable": available,
    }


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> Any:
    global _loaded_model_name
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    blocks, filename = MODELS[model_name]
    path = _weights_dir() / filename
    if not path.is_file():
        raise FileNotFoundError("configured upscale model mount is unavailable")
    network = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=blocks, num_grow_ch=32, scale=4
    )
    backend = RealESRGANer(
        scale=4,
        model_path=str(path),
        model=network,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=torch.device("cuda"),
    )
    _loaded_model_name = model_name
    return backend


def _release_resident_model() -> None:
    global _loaded_model_name
    with _model_lock:
        _load_model.cache_clear()
        _loaded_model_name = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                if callable(ipc_collect):
                    ipc_collect()
        except ImportError:
            pass
        except (AttributeError, RuntimeError) as exc:
            logger.exception("upscale CUDA cache release failed")
            raise RuntimeError("upscale CUDA cache release failed") from exc


def _release_model_for_transition(requested_model: str) -> None:
    if _loaded_model_name is None or _loaded_model_name == requested_model:
        return
    _release_resident_model()


def _shutdown_unload() -> None:
    try:
        _release_resident_model()
    except Exception:
        logger.exception("upscale worker shutdown unload failed")


atexit.register(_shutdown_unload)


def start_durable_runner() -> None:
    if os.getenv("IMAGE_API_ENABLE_PROCESSING_RUNNER", "false").lower() != "true":
        return
    settings = Settings.from_env()
    store = TaskStore(settings.database_path, settings.max_queue_depth)
    recovered = recover_processing_tasks(
        "upscale", store, settings.output_dir, settings.source_dir, settings
    )
    if recovered:
        logger.warning("reconciled interrupted upscale tasks: count=%s", recovered)

    def model(source: Path, request: dict[str, object]) -> bytes:
        with source.open("rb") as handle:
            data = handle.read(settings.processing_max_upload_bytes + 1)
        if len(data) > settings.processing_max_upload_bytes:
            raise ValueError("persisted source exceeds configured limit")
        model_name = request.get("model")
        outscale = request.get("outscale")
        tile = request.get("tile")
        if (
            not isinstance(model_name, str)
            or model_name not in MODELS
            or not isinstance(outscale, (int, float))
            or isinstance(outscale, bool)
            or type(tile) is not int
        ):
            raise ValueError("invalid persisted upscale parameters")
        with _model_lock:
            return _run_upscale(
                data,
                model=model_name,
                outscale=float(outscale),
                tile=tile,
            )

    runner = ProcessingRunner(
        "upscale",
        store,
        GpuLane(settings.gpu_lane_path, settings.lane_timeout_seconds),
        settings.source_dir,
        settings.output_dir,
        model,
        settings,
        peer_evictor=PeerEvictor(
            (
                os.getenv("IMAGE_API_BACKGROUND_WORKER_URL", "http://background-worker:9002"),
                os.getenv("IMAGE_API_GENERATION_WORKER_URL", "http://generation-worker:9003"),
            )
        ),
    )

    def loop() -> None:
        poll = float(os.getenv("IMAGE_API_PROCESSING_POLL_SECONDS", "0.5"))
        while True:
            if not runner.run_one():
                time.sleep(poll)

    threading.Thread(target=loop, name="upscale-task-runner", daemon=True).start()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    start_durable_runner()
    yield


app = FastAPI(title="image-api-upscale-worker", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    return _runtime_status()


@app.post("/internal/unload")
def unload() -> dict[str, object]:
    try:
        _release_resident_model()
    except Exception as exc:
        logger.exception("upscale worker unload failed")
        raise HTTPException(500, "model unload failed") from exc
    return {"unloaded": True, "loaded": False}


def _validate_worker_dimensions(width: int, height: int, *, output: bool) -> None:
    suffix = "OUTPUT" if output else "INPUT"
    validate_dimensions(
        width,
        height,
        max_width=int(os.getenv("IMAGE_API_PROCESSING_MAX_INPUT_WIDTH", "8192")),
        max_height=int(os.getenv("IMAGE_API_PROCESSING_MAX_INPUT_HEIGHT", "8192")),
        max_pixels=int(os.getenv(f"IMAGE_API_PROCESSING_MAX_{suffix}_PIXELS", "67108864")),
        max_decoded_bytes=int(
            os.getenv(f"IMAGE_API_PROCESSING_MAX_DECODED_{suffix}_BYTES", "268435456")
        ),
    )


def _validate_native_processing(width: int, height: int) -> None:
    native_width = width * 4
    native_height = height * 4
    native_pixels = native_width * native_height
    if (
        native_width > int(os.getenv("IMAGE_API_PROCESSING_MAX_NATIVE_WIDTH", "16384"))
        or native_height > int(os.getenv("IMAGE_API_PROCESSING_MAX_NATIVE_HEIGHT", "16384"))
        or native_pixels > int(os.getenv("IMAGE_API_PROCESSING_MAX_NATIVE_PIXELS", "268435456"))
        or native_pixels * 3 * 4
        > int(os.getenv("IMAGE_API_PROCESSING_MAX_NATIVE_BYTES", "3221225472"))
    ):
        raise ValueError("Real-ESRGAN native processing exceeds configured limit")


def _load_rgb_source(data: bytes) -> tuple[Image.Image, tuple[int, int]]:
    with Image.open(io.BytesIO(data)) as image:
        _validate_worker_dimensions(image.width, image.height, output=False)
        _validate_native_processing(image.width, image.height)
        image.load()
        return image.convert("RGB"), image.size


def _effective_tile(tile: int, width: int, height: int) -> int:
    if tile < 0 or tile > 1024 or (tile and tile % 32):
        raise ValueError("invalid tile")
    return 512 if tile == 0 and max(width, height) > 1024 else tile


def _run_upscale(data: bytes, *, model: str, outscale: float, tile: int) -> bytes:
    import numpy as np
    import torch

    if model not in MODELS:
        raise ValueError("unsupported upscale model")
    if not math.isfinite(outscale) or not 1 <= outscale <= 4:
        raise ValueError("invalid outscale")
    rgb_image, source_size = _load_rgb_source(data)
    expected_size = (round(source_size[0] * outscale), round(source_size[1] * outscale))
    _validate_worker_dimensions(*expected_size, output=True)
    source = np.asarray(rgb_image)[:, :, ::-1]
    _release_model_for_transition(model)
    backend = _load_model(model)
    previous = backend.tile_size
    backend.tile_size = _effective_tile(tile, source_size[0], source_size[1])
    try:
        with torch.inference_mode():
            result, _ = backend.enhance(source, outscale=outscale)
    finally:
        backend.tile_size = previous
    import cv2

    if result.ndim != 3 or result.shape[2] != 3:
        raise RuntimeError("upscale backend returned non-RGB output")
    if (result.shape[1], result.shape[0]) != expected_size:
        raise RuntimeError("upscale backend returned unexpected dimensions")
    ok, encoded = cv2.imencode(".png", result)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return bytes(encoded.tobytes())


@app.post("/internal/upscale", response_class=Response)
async def upscale(
    file: Annotated[UploadFile, File()],
    model: Annotated[Literal["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], Query()],
    outscale: Annotated[float, Query(ge=1, le=4)],
    tile: Annotated[int, Query(ge=0, le=1024)],
) -> Response:
    if tile and tile % 32:
        raise HTTPException(422, "invalid tile")
    max_upload_bytes = int(os.getenv("IMAGE_API_PROCESSING_MAX_UPLOAD_BYTES", "280000000"))
    data = await read_bounded_upload(file, max_upload_bytes)
    try:

        def operation() -> bytes:
            PeerEvictor(
                (
                    os.getenv("IMAGE_API_BACKGROUND_WORKER_URL", "http://background-worker:9002"),
                    os.getenv("IMAGE_API_GENERATION_WORKER_URL", "http://generation-worker:9003"),
                )
            )()
            with _model_lock:
                return _run_upscale(data, model=model, outscale=outscale, tile=tile)

        encoded = await asyncio.to_thread(
            execute_in_gpu_lane,
            "upscale",
            operation,
        )
        max_output_bytes = int(
            os.getenv("IMAGE_API_PROCESSING_MAX_ENCODED_OUTPUT_BYTES", "300000000")
        )
        if len(encoded) > max_output_bytes:
            raise RuntimeError("encoded upscale output exceeds configured limit")
        return Response(encoded, media_type="image/png")
    except Exception as exc:
        logger.exception("upscale worker failed: model=%s", model)
        raise HTTPException(500, "internal image processing error") from exc
