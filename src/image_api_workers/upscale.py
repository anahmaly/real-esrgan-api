from __future__ import annotations

import asyncio
import atexit
import gc
import io
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image

from image_api.workers import PeerEvictor
from image_api_workers.execution import execute_in_gpu_lane

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


app = FastAPI(title="image-api-upscale-worker", docs_url=None, redoc_url=None)


def _shutdown_unload() -> None:
    try:
        _release_resident_model()
    except Exception:
        logger.exception("upscale worker shutdown unload failed")


atexit.register(_shutdown_unload)


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


def _run_upscale(data: bytes, *, model: str, outscale: float, tile: int) -> bytes:
    import numpy as np
    import torch

    with Image.open(io.BytesIO(data)) as image:
        image.load()
        source = np.asarray(image.convert("RGBA" if image.mode == "RGBA" else "RGB"))
    if source.shape[-1] == 3:
        source = source[:, :, ::-1]
    else:
        source = source[:, :, [2, 1, 0, 3]]
    _release_model_for_transition(model)
    backend = _load_model(model)
    previous = backend.tile_size
    backend.tile_size = tile
    try:
        with torch.inference_mode():
            result, _ = backend.enhance(source, outscale=outscale)
    finally:
        backend.tile_size = previous
    import cv2

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
    data = await file.read()
    await file.close()
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
        return Response(encoded, media_type="image/png")
    except Exception as exc:
        logger.exception("upscale worker failed: model=%s", model)
        raise HTTPException(500, "internal image processing error") from exc
