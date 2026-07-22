from __future__ import annotations

import asyncio
import atexit
import gc
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
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
_active_model: str | None = None
_model_lock = threading.RLock()


def _birefnet_config() -> Any:
    from rembg_api.birefnet_hr import BiRefNetConfig, DEFAULT_REVISION

    return BiRefNetConfig(
        source=os.getenv("IMAGE_API_BIREFNET_WEIGHTS_PATH", "/models/birefnet-hr"),
        revision=os.getenv("IMAGE_API_BIREFNET_REVISION", DEFAULT_REVISION),
        local_files_only=True,
        trust_remote_code=True,
        cache_dir=None,
        device="cuda",
        precision="fp16",
        inference_size=2048,
        foreground_refinement=False,
        max_concurrency=1,
    )


def _release_resident_models() -> None:
    global _active_model
    with _model_lock:
        release_errors: list[BaseException] = []
        try:
            from rembg_api.birefnet_hr import clear_cache

            clear_cache()
        except ImportError:
            pass
        except (AttributeError, RuntimeError) as exc:
            release_errors.append(exc)
            logger.exception("BiRefNet cache release failed")
        try:
            from rembg_api.bria_rmbg import clear_bria_backend_cache

            clear_bria_backend_cache(release_cuda_cache=True)
        except ImportError:
            pass
        except (AttributeError, RuntimeError) as exc:
            release_errors.append(exc)
            logger.exception("BRIA cache release failed")
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
            release_errors.append(exc)
            logger.exception("background CUDA cache release failed")
        _active_model = None
        if release_errors:
            raise RuntimeError("background model release failed") from release_errors[0]


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


def _run_background(
    data: bytes,
    *,
    model: str,
    alpha_blur: float,
    alpha_erode: int,
    alpha_dilate: int,
    alpha_threshold: int,
    birefnet_inference_size: int,
    birefnet_foreground_refinement: bool,
    model_input_size: int,
    despill_enabled: bool = False,
    despill_color: str = "black",
    despill_hex_color: str = "000000",
) -> bytes:
    global _active_model
    if not 512 <= birefnet_inference_size <= 4096:
        raise ValueError("invalid BiRefNet inference size")
    with Image.open(io.BytesIO(data)) as source_image:
        expected_size = source_image.size
        _validate_worker_dimensions(*expected_size, output=False)
        _validate_worker_dimensions(*expected_size, output=True)
        source_image.verify()
    if _active_model is not None and _active_model != model:
        _release_resident_models()
    if model == "birefnet-hr-matting":
        from rembg_api.birefnet_hr import remove_with_birefnet

        removed = remove_with_birefnet(
            data,
            inference_size=birefnet_inference_size,
            foreground_refinement=birefnet_foreground_refinement,
            config=_birefnet_config(),
        )
    elif model == "bria-rmbg-2.0":
        from rembg_api.bria_rmbg import remove_with_bria_rmbg_2

        removed = remove_with_bria_rmbg_2(
            data,
            model_input_size=model_input_size,
            device="cuda",
            dtype="fp16",
            model_path=os.getenv("IMAGE_API_BRIA_WEIGHTS_PATH", "/models/bria-rmbg-2.0"),
        )
    else:
        raise ValueError("unsupported background-removal model")
    if not isinstance(removed, bytes):
        raise RuntimeError("background backend returned invalid bytes")
    from rembg_api.image_processing import AlphaOptions, DespillOptions, process_png_bytes

    encoded = process_png_bytes(
        removed,
        alpha=AlphaOptions(
            blur=alpha_blur,
            erode=alpha_erode,
            dilate=alpha_dilate,
            threshold=alpha_threshold,
        ),
        despill=DespillOptions(
            enabled=despill_enabled,
            color=despill_color,
            hex_color=despill_hex_color,
        ),
        background_color="transparent",
        background_hex="ffffff",
        return_alpha=False,
        return_checker_preview=False,
        checker_size=32,
        max_encoded_bytes=int(
            os.getenv("IMAGE_API_PROCESSING_MAX_ENCODED_OUTPUT_BYTES", "300000000")
        ),
    )
    if not isinstance(encoded, bytes):
        raise RuntimeError("background post-processing returned invalid bytes")
    with Image.open(io.BytesIO(encoded)) as output:
        output.load()
        if output.mode != "RGBA":
            raise RuntimeError("background backend did not return RGBA")
        if output.size != expected_size:
            raise RuntimeError("background backend returned unexpected dimensions")
    _active_model = model
    return encoded


def _health() -> dict[str, object]:
    bria = Path(os.getenv("IMAGE_API_BRIA_WEIGHTS_PATH", "/models/bria-rmbg-2.0"))
    birefnet = Path(os.getenv("IMAGE_API_BIREFNET_WEIGHTS_PATH", "/models/birefnet-hr"))
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception as exc:
        logger.warning(
            "background CUDA runtime probe failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        cuda = False
    loaded = _active_model is not None
    mounts = bria.is_dir() and birefnet.is_dir()
    return {
        "ready": cuda and mounts,
        "loaded": bool(loaded),
        "loadedModel": _active_model,
        "device": "cuda" if cuda else "unavailable",
        "weightsAvailable": mounts,
    }


def _shutdown_unload() -> None:
    try:
        _release_resident_models()
    except Exception:
        logger.exception("background worker shutdown unload failed")


atexit.register(_shutdown_unload)


def start_durable_runner() -> None:
    if os.getenv("IMAGE_API_ENABLE_PROCESSING_RUNNER", "false").lower() != "true":
        return
    settings = Settings.from_env()
    store = TaskStore(settings.database_path, settings.max_queue_depth)
    recovered = recover_processing_tasks(
        "background-removal", store, settings.output_dir, settings.source_dir, settings
    )
    if recovered:
        logger.warning("reconciled interrupted background tasks: count=%s", recovered)

    def model(source: Path, request: dict[str, object]) -> bytes:
        with source.open("rb") as handle:
            data = handle.read(settings.processing_max_upload_bytes + 1)
        if len(data) > settings.processing_max_upload_bytes:
            raise ValueError("persisted source exceeds configured limit")
        model_name = request.get("model")
        alpha_blur = request.get("alpha_blur")
        alpha_erode = request.get("alpha_erode")
        alpha_dilate = request.get("alpha_dilate")
        alpha_threshold = request.get("alpha_threshold")
        inference_size = request.get("birefnet_inference_size")
        refinement = request.get("birefnet_foreground_refinement")
        model_input_size = request.get("model_input_size")
        despill_enabled = request.get("despill_enabled")
        despill_color = request.get("despill_color")
        despill_hex_color = request.get("despill_hex_color")
        if (
            model_name not in {"bria-rmbg-2.0", "birefnet-hr-matting"}
            or not isinstance(model_name, str)
            or not isinstance(alpha_blur, (int, float))
            or isinstance(alpha_blur, bool)
            or any(
                type(value) is not int
                for value in (
                    alpha_erode,
                    alpha_dilate,
                    alpha_threshold,
                    inference_size,
                    model_input_size,
                )
            )
            or type(refinement) is not bool
            or type(despill_enabled) is not bool
            or despill_color not in {"black", "white", "green", "blue", "custom"}
            or not isinstance(despill_color, str)
            or not isinstance(despill_hex_color, str)
        ):
            raise ValueError("invalid persisted background-removal parameters")
        assert type(alpha_erode) is int
        assert type(alpha_dilate) is int
        assert type(alpha_threshold) is int
        assert type(inference_size) is int
        assert type(model_input_size) is int
        assert type(refinement) is bool
        assert type(despill_enabled) is bool
        with _model_lock:
            return _run_background(
                data,
                model=model_name,
                alpha_blur=float(alpha_blur),
                alpha_erode=alpha_erode,
                alpha_dilate=alpha_dilate,
                alpha_threshold=alpha_threshold,
                birefnet_inference_size=inference_size,
                birefnet_foreground_refinement=refinement,
                model_input_size=model_input_size,
                despill_enabled=despill_enabled,
                despill_color=despill_color,
                despill_hex_color=despill_hex_color,
            )

    runner = ProcessingRunner(
        "background-removal",
        store,
        GpuLane(settings.gpu_lane_path, settings.lane_timeout_seconds),
        settings.source_dir,
        settings.output_dir,
        model,
        settings,
        peer_evictor=PeerEvictor(
            (
                os.getenv("IMAGE_API_UPSCALE_WORKER_URL", "http://upscale-worker:9001"),
                os.getenv("IMAGE_API_GENERATION_WORKER_URL", "http://generation-worker:9003"),
            )
        ),
    )

    def loop() -> None:
        poll = float(os.getenv("IMAGE_API_PROCESSING_POLL_SECONDS", "0.5"))
        while True:
            if not runner.run_one():
                time.sleep(poll)

    threading.Thread(target=loop, name="background-task-runner", daemon=True).start()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    start_durable_runner()
    yield


app = FastAPI(title="image-api-background-worker", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    return _health()


@app.post("/internal/unload")
def unload() -> dict[str, object]:
    try:
        _release_resident_models()
    except Exception as exc:
        logger.exception("background worker unload failed")
        raise HTTPException(500, "model unload failed") from exc
    return {"unloaded": True, "loaded": False}


@app.post("/internal/background-removal", response_class=Response)
async def remove_background(
    file: Annotated[UploadFile, File()],
    model: Annotated[
        Literal["bria-rmbg-2.0", "birefnet-hr-matting"],
        Query(),
    ],
    alpha_blur: Annotated[float, Query(ge=0, le=20)] = 0,
    alpha_erode: Annotated[int, Query(ge=0, le=100)] = 0,
    alpha_dilate: Annotated[int, Query(ge=0, le=100)] = 0,
    alpha_threshold: Annotated[int, Query(ge=0, le=255)] = 0,
    birefnet_inference_size: Annotated[int, Query(ge=512, le=4096)] = 2048,
    birefnet_foreground_refinement: bool = False,
    model_input_size: Annotated[int, Query(ge=512, le=2048)] = 1024,
) -> Response:
    max_upload_bytes = int(os.getenv("IMAGE_API_PROCESSING_MAX_UPLOAD_BYTES", "280000000"))
    data = await read_bounded_upload(file, max_upload_bytes)
    try:

        def operation() -> bytes:
            PeerEvictor(
                (
                    os.getenv("IMAGE_API_UPSCALE_WORKER_URL", "http://upscale-worker:9001"),
                    os.getenv("IMAGE_API_GENERATION_WORKER_URL", "http://generation-worker:9003"),
                )
            )()
            with _model_lock:
                return _run_background(
                    data,
                    model=model,
                    alpha_blur=alpha_blur,
                    alpha_erode=alpha_erode,
                    alpha_dilate=alpha_dilate,
                    alpha_threshold=alpha_threshold,
                    birefnet_inference_size=birefnet_inference_size,
                    birefnet_foreground_refinement=birefnet_foreground_refinement,
                    model_input_size=model_input_size,
                )

        encoded = await asyncio.to_thread(
            execute_in_gpu_lane,
            "background-removal",
            operation,
        )
        return Response(encoded, media_type="image/png")
    except Exception as exc:
        logger.exception("background worker failed: model=%s", model)
        raise HTTPException(500, "internal image processing error") from exc
