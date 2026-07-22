from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import threading
import uuid
from tempfile import SpooledTemporaryFile
from typing import Annotated, Any, Literal, cast

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from image_api.config import Settings, ideogram_weights_available, longcat_weights_available
from image_api.generation import source_file_lock, worker_heartbeat_alive
from image_api.images import (
    ImageInfo,
    ImageTooLarge,
    InvalidImage,
    InvalidWorkerImage,
    processing_output_size,
    validate_image,
    validate_png_output,
)
from image_api.lane import GpuLane, LaneBusy
from image_api.processing import validate_processing_output
from image_api.state import state_write_ready
from image_api.store import IdempotencyConflict, QueueFull, TaskKind, TaskRecord, TaskStore
from image_api.workers import HttpWorkerClient, WorkerClient, WorkerUnavailable

logger = logging.getLogger(__name__)

UPSCALE_MODELS = ("RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B")
BACKGROUND_MODELS = ("bria-rmbg-2.0", "birefnet-hr-matting")
SAMPLER_PRESETS = ("V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12")
LONGCAT_MODELS = ("longcat-image-edit", "longcat-image-edit-turbo")
TASK_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
IDEMPOTENCY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
UPLOAD_CHUNK_BYTES = 64 * 1024
UPLOAD_SPOOL_MEMORY_BYTES = 8 * 1024 * 1024
_ADMISSION_LOCK = threading.Lock()


class RequestBodyTooLarge(BaseException):
    pass


class RequestBodyLimitMiddleware:
    def __init__(
        self,
        app: Any,
        default_max_bytes: int,
        route_max_bytes: dict[str, int],
    ) -> None:
        self.app = app
        self.default_max_bytes = default_max_bytes
        self.route_max_bytes = route_max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        path = scope.get("path")
        max_bytes = (
            self.route_max_bytes.get(path, self.default_max_bytes)
            if isinstance(path, str)
            else self.default_max_bytes
        )
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared is not None:
            if not declared.isdigit():
                await JSONResponse(
                    {"error": {"code": "invalid_request", "message": "Invalid request"}},
                    status_code=400,
                )(scope, receive, send)
                return
            if int(declared) > max_bytes:
                await JSONResponse(
                    {"error": {"code": "request_too_large", "message": "Request is too large"}},
                    status_code=413,
                )(scope, receive, send)
                return
        consumed = 0
        started = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal consumed
            message = cast(dict[str, Any], await receive())
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > max_bytes:
                    raise RequestBodyTooLarge
            return message

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if started:
                raise RuntimeError("request limit exceeded after response start")
            await JSONResponse(
                {"error": {"code": "request_too_large", "message": "Request is too large"}},
                status_code=413,
            )(scope, receive, send)


class GenerationRequest(BaseModel):
    width: int = Field(ge=256, le=2048, multiple_of=16)
    height: int = Field(ge=256, le=2048, multiple_of=16)
    seed: int = Field(ge=0, le=2**32 - 1)
    sampler_preset: Literal["V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12"]
    structured_caption: dict[str, Any] | None = None
    prompt: str | None = Field(default=None, min_length=1, max_length=4000)
    magic_prompt: bool = False

    @model_validator(mode="after")
    def validate_caption_mode(self) -> "GenerationRequest":
        structured = self.structured_caption is not None
        plain = self.prompt is not None
        if structured == plain:
            raise ValueError("provide exactly one caption mode")
        if structured:
            if not self.structured_caption:
                raise ValueError("structured_caption must be a non-empty JSON object")
            encoded = json.dumps(self.structured_caption, sort_keys=True, separators=(",", ":"))
            if len(encoded.encode("utf-8")) > 64_000:
                raise ValueError("structured_caption is too large")
            if self.magic_prompt:
                raise ValueError("magic_prompt is only valid with prompt")
        elif not self.magic_prompt:
            raise ValueError("plain prompts require magic_prompt=true")
        return self


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    total = 0
    try:
        with SpooledTemporaryFile(max_size=UPLOAD_SPOOL_MEMORY_BYTES, mode="w+b") as spool:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                if len(chunk) > max_bytes - total:
                    raise ImageTooLarge("upload exceeds configured limit")
                spool.write(chunk)
                total += len(chunk)
            spool.seek(0)
            return spool.read()
    finally:
        await file.close()


async def _validated_sync_upload(file: UploadFile, settings: Settings) -> ImageInfo:
    try:
        await file.seek(0)
        info = validate_image(
            file.file,
            max_bytes=settings.processing_max_upload_bytes,
            max_width=settings.processing_max_input_width,
            max_height=settings.processing_max_input_height,
            max_pixels=settings.processing_max_input_pixels,
            max_decoded_bytes=settings.processing_max_decoded_input_bytes,
        )
        await file.seek(0)
        return info
    except BaseException:
        await file.close()
        raise


def _worker_image_chunks(stream: Any) -> Any:
    try:
        while chunk := stream.read(UPLOAD_CHUNK_BYTES):
            yield chunk
    finally:
        stream.close()


def _worker_image_response(encoded: object) -> Response:
    if isinstance(encoded, bytes):
        return Response(encoded, media_type="image/png")
    if not all(hasattr(encoded, member) for member in ("read", "seek", "close")):
        raise InvalidWorkerImage("worker output type mismatch")
    stream = cast(Any, encoded)
    try:
        stream.seek(0)
    except BaseException:
        stream.close()
        raise

    return StreamingResponse(
        _worker_image_chunks(stream),
        media_type="image/png",
        background=BackgroundTask(stream.close),
    )


def _normalize_source(data: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise InvalidImage("unsupported source image format")
            if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
                raise InvalidImage("unsupported source image mode")
            normalized = image.convert("RGB")
    except InvalidImage:
        raise
    except (OSError, ValueError) as exc:
        raise InvalidImage("source image could not be normalized") from exc
    output = io.BytesIO()
    normalized.save(output, "PNG", compress_level=9, optimize=False)
    return output.getvalue()


def _persist_source_atomically(
    source_dir: os.PathLike[str], data: bytes, original_hash: str
) -> str:
    directory = os.fspath(source_dir)
    os.makedirs(directory, exist_ok=True)
    normalized_hash = hashlib.sha256(data).hexdigest()
    name = f"{original_hash}-{normalized_hash}.png"
    final_path = os.path.join(directory, name)
    if os.path.isfile(final_path):
        with open(final_path, "rb") as handle:
            if handle.read(len(data) + 1) != data:
                raise OSError("persisted source content mismatch")
        return name
    temporary = os.path.join(directory, f".{name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final_path)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return name


def _remove_orphan_source(source_dir: os.PathLike[str], name: str, store: TaskStore) -> None:
    if store.source_referenced(name):
        return
    path = os.path.join(os.fspath(source_dir), name)
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.error(
            "orphan source cleanup failed: source_name=%s",
            name,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return
    try:
        directory_fd = os.open(os.fspath(source_dir), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        logger.error(
            "orphan source directory sync failed: source_name=%s",
            name,
            exc_info=(type(exc), exc, exc.__traceback__),
        )


def _safe_task(task: TaskRecord) -> dict[str, object]:
    result: dict[str, object] = {
        "taskId": task.task_id,
        "status": task.status,
    }
    if task.request.get("task_type") == "image-edit":
        result.update({"model": task.request["model"], "seed": task.request["seed"]})
    else:
        result.update(
            {
                "width": task.request["width"],
                "height": task.request["height"],
                "seed": task.request["seed"],
                "samplerPreset": task.request["sampler_preset"],
            }
        )
    if task.error_code:
        is_edit = task.request.get("task_type") == "image-edit"
        result["error"] = {
            "code": task.error_code,
            "message": ("Image edit did not complete" if is_edit else "Generation did not complete")
            if task.status == "failed"
            else ("Image edit unavailable" if is_edit else "Generation unavailable"),
        }
    return result


def _safe_processing_task(task: TaskRecord) -> dict[str, object]:
    result: dict[str, object] = {
        "taskId": task.task_id,
        "status": task.status,
        "capability": task.task_kind,
        "model": task.request["model"],
        "sourceSha256": task.request["source_image_sha256"],
        "requestedWidth": task.request["requested_width"],
        "requestedHeight": task.request["requested_height"],
        "expectedWidth": task.request["expected_width"],
        "expectedHeight": task.request["expected_height"],
        "expectedMode": task.request["expected_mode"],
    }
    if task.status == "succeeded":
        result["output"] = {
            "fileName": task.image_name,
            "sha256": task.output_sha256,
            "width": task.output_width,
            "height": task.output_height,
            "mode": task.output_mode,
        }
    if task.error_code:
        result["error"] = {
            "code": task.error_code,
            "message": "Image processing did not complete",
        }
    return result


async def _persist_processing_upload(
    file: UploadFile, settings: Settings
) -> tuple[str, str, ImageInfo, bool]:
    settings.source_dir.mkdir(parents=True, exist_ok=True)
    temporary = settings.source_dir / f".processing-upload.{uuid.uuid4().hex}.tmp"
    digest = hashlib.sha256()
    total = 0
    try:
        with temporary.open("xb") as target:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                total += len(chunk)
                if total > settings.processing_max_upload_bytes:
                    raise ImageTooLarge("upload exceeds configured limit")
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        with temporary.open("rb") as handle:
            info = validate_image(
                handle,
                max_bytes=settings.processing_max_upload_bytes,
                max_width=settings.processing_max_input_width,
                max_height=settings.processing_max_input_height,
                max_pixels=settings.processing_max_input_pixels,
                max_decoded_bytes=settings.processing_max_decoded_input_bytes,
            )
        source_hash = digest.hexdigest()
        source_name = f"{source_hash}-{source_hash}.png"
        final_path = settings.source_dir / source_name
        existed = final_path.is_file()
        if existed:
            with final_path.open("rb") as existing:
                existing_hash = hashlib.file_digest(existing, "sha256").hexdigest()
            if existing_hash != source_hash:
                raise OSError("persisted source content mismatch")
        else:
            os.replace(temporary, final_path)
            directory_fd = os.open(settings.source_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return source_name, source_hash, info, existed
    finally:
        temporary.unlink(missing_ok=True)
        await file.close()


def _generation_health(
    settings: Settings, worker_status: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    repository_id = os.getenv("IMAGE_API_IDEOGRAM_REPOSITORY_ID", "ideogram-ai/ideogram-4-nf4")
    mounted = {
        "ideogram-4-nf4": settings.generation_test_mode
        or ideogram_weights_available(settings.ideogram_weights_path, repository_id),
        "longcat-image-edit": settings.generation_test_mode
        or longcat_weights_available(
            settings.longcat_edit_weights_path, settings.longcat_edit_revision
        ),
        "longcat-image-edit-turbo": settings.generation_test_mode
        or longcat_weights_available(
            settings.longcat_edit_turbo_weights_path, settings.longcat_edit_turbo_revision
        ),
    }
    worker_available = settings.generation_test_mode or (
        bool(worker_status.get("workerReachable"))
        and worker_heartbeat_alive(
            settings.generation_heartbeat_path,
            max_age_seconds=settings.generation_heartbeat_max_age_seconds,
        )
    )
    loaded_model = worker_status.get("loadedModel")
    if loaded_model not in {"ideogram-4-nf4", *LONGCAT_MODELS}:
        loaded_model = None
    raw_device = worker_status.get("device")
    device = (
        "cpu-test"
        if settings.generation_test_mode
        else raw_device
        if raw_device in {"cuda", "unavailable"}
        else "unavailable"
    )
    cuda_available = settings.generation_test_mode or device == "cuda"
    raw_models = worker_status.get("models")
    worker_models = raw_models if isinstance(raw_models, dict) else None

    def model_weights_ready(model: str) -> bool:
        if not mounted[model]:
            return False
        if worker_models is None:
            return settings.generation_test_mode or bool(worker_status.get("ready"))
        model_status = worker_models.get(model)
        return isinstance(model_status, dict) and bool(model_status.get("weightsAvailable"))

    def status(models: tuple[str, ...]) -> dict[str, object]:
        model_weights = {model: model_weights_ready(model) for model in models}
        model_ready = {
            model: worker_available and cuda_available and available
            for model, available in model_weights.items()
        }
        ready = all(model_ready.values())
        reason = None
        if not all(model_weights.values()):
            reason = "weights_unavailable"
        elif not worker_available:
            reason = "worker_unavailable"
        elif not cuda_available:
            reason = "cuda_unavailable"
        result: dict[str, object] = {
            "ready": ready,
            "loaded": loaded_model in models,
            "device": device,
            "weightsAvailable": all(model_weights.values()),
            "workerAvailable": worker_available,
            "reason": None if ready else reason,
            "models": {
                model: {
                    "ready": model_ready[model],
                    "weightsAvailable": model_weights[model],
                    "loaded": loaded_model == model,
                }
                for model in models
            },
        }
        if loaded_model in models:
            result["loadedModel"] = loaded_model
        return result

    generation = status(("ideogram-4-nf4",))
    generation["quantization"] = "nf4"
    return generation, status(LONGCAT_MODELS)


def create_app(
    *,
    settings: Settings | None = None,
    store: TaskStore | None = None,
    workers: WorkerClient | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.source_dir.mkdir(parents=True, exist_ok=True)
    store = store or TaskStore(settings.database_path, settings.max_queue_depth)
    workers = workers or HttpWorkerClient(
        os.getenv("IMAGE_API_UPSCALE_WORKER_URL", "http://upscale-worker:9001"),
        os.getenv("IMAGE_API_BACKGROUND_WORKER_URL", "http://background-worker:9002"),
        settings.worker_timeout_seconds,
        settings.processing_max_encoded_output_bytes,
        generation_url=os.getenv(
            "IMAGE_API_GENERATION_WORKER_URL", "http://generation-worker:9003"
        ),
    )
    lane = GpuLane(settings.gpu_lane_path, settings.lane_timeout_seconds)

    app = FastAPI(title="image-api", version="1.0.0")
    app.add_middleware(
        RequestBodyLimitMiddleware,
        default_max_bytes=settings.max_request_bytes,
        route_max_bytes={
            "/v1/upscale": settings.processing_max_request_bytes,
            "/v1/background-removal": settings.processing_max_request_bytes,
            "/v1/upscale-tasks": settings.processing_max_request_bytes,
            "/v1/background-removal-tasks": settings.processing_max_request_bytes,
        },
    )

    @app.exception_handler(WorkerUnavailable)
    async def worker_unavailable(_: Request, exc: WorkerUnavailable) -> JSONResponse:
        logger.error(
            "capability worker unavailable",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "worker_unavailable",
                    "message": "Image capability is temporarily unavailable",
                }
            },
        )

    @app.exception_handler(LaneBusy)
    async def lane_busy(_: Request, exc: LaneBusy) -> JSONResponse:
        logger.warning(
            "GPU lane admission rejected",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": {"code": "gpu_lane_busy", "message": "Image processing capacity is busy"}
            },
        )

    @app.exception_handler(ImageTooLarge)
    async def image_too_large(_: Request, exc: ImageTooLarge) -> JSONResponse:
        logger.warning(
            "image admission rejected by configured limit",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            {"error": {"code": "image_too_large", "message": "Image exceeds accepted limits"}},
            status_code=413,
        )

    @app.exception_handler(InvalidImage)
    async def invalid_image(_: Request, exc: InvalidImage) -> JSONResponse:
        logger.warning(
            "image admission rejected invalid input",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            {"error": {"code": "invalid_image", "message": "Uploaded file is not a valid image"}},
            status_code=400,
        )

    @app.exception_handler(InvalidWorkerImage)
    async def invalid_worker_image(_: Request, exc: InvalidWorkerImage) -> JSONResponse:
        logger.error(
            "capability worker returned invalid output",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            {
                "error": {
                    "code": "invalid_worker_output",
                    "message": "Image capability returned invalid output",
                }
            },
            status_code=502,
        )

    @app.get("/health")
    def health() -> dict[str, object]:
        raw_status = workers.health()
        capability_status: dict[str, dict[str, object]] = {}
        for capability in ("upscale", "background-removal"):
            worker_status = raw_status.get(capability, {})
            device = worker_status.get("device")
            if device not in {"cuda", "cpu-test", "unavailable"}:
                device = "unavailable"
            capability_status[capability] = {
                "ready": bool(worker_status.get("ready")),
                "loaded": bool(worker_status.get("loaded")),
                "device": device,
            }
            if "weightsAvailable" in worker_status:
                capability_status[capability]["weightsAvailable"] = bool(
                    worker_status["weightsAvailable"]
                )
            allowed_models = UPSCALE_MODELS if capability == "upscale" else BACKGROUND_MODELS
            loaded_model = worker_status.get("loadedModel")
            if loaded_model in allowed_models:
                capability_status[capability]["loadedModel"] = loaded_model
        generation_status, editing_status = _generation_health(
            settings, raw_status.get("generation", {})
        )
        capability_status["generation"] = generation_status
        capability_status["image-editing"] = editing_status
        state_ready = state_write_ready(
            settings.state_dir, settings.database_path, settings.source_dir
        )
        return {
            "service": "image-api",
            "status": "ok"
            if state_ready and all(bool(v.get("ready")) for v in capability_status.values())
            else "degraded",
            "capabilities": capability_status,
            "state": {"ready": state_ready},
            "gpuLane": lane.status(),
        }

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        return {
            "models": [
                *({"capability": "upscale", "model": model} for model in UPSCALE_MODELS),
                *(
                    {"capability": "background-removal", "model": model}
                    for model in BACKGROUND_MODELS
                ),
                {
                    "capability": "generation",
                    "model": "ideogram-4-nf4",
                    "acceptsSourceImage": False,
                    "samplerPresets": list(SAMPLER_PRESETS),
                    "dimensions": {"minimum": 256, "maximum": 2048, "multipleOf": 16},
                },
                {
                    "capability": "image-editing",
                    "model": "longcat-image-edit",
                    "acceptsSourceImage": True,
                    "inputImages": 1,
                    "defaults": {"guidanceScale": 4.5, "steps": 50},
                },
                {
                    "capability": "image-editing",
                    "model": "longcat-image-edit-turbo",
                    "acceptsSourceImage": True,
                    "inputImages": 1,
                    "defaults": {"guidanceScale": 1.0, "steps": 8},
                },
            ]
        }

    @app.post("/v1/models/unload")
    def unload_models() -> JSONResponse:
        with lane.acquire("model-unload"):
            results = workers.unload_all()
        complete = all(bool(value.get("unloaded")) for value in results.values())
        return JSONResponse(
            {"unloaded": complete, "workers": results},
            status_code=200 if complete else 503,
        )

    @app.post("/v1/upscale-tasks", status_code=202)
    async def admit_upscale_task(
        file: Annotated[UploadFile, File()],
        model: Annotated[Literal["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], Query()],
        outscale: Annotated[float, Query(ge=1, le=4)],
        tile: Annotated[int, Query(ge=0, le=1024)],
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
    ) -> dict[str, object]:
        if not IDEMPOTENCY_PATTERN.fullmatch(idempotency_key):
            await file.close()
            raise HTTPException(422, "invalid idempotency key")
        if tile != 0 and tile % 32:
            await file.close()
            raise HTTPException(422, "tile must be zero or a multiple of 32")
        with _ADMISSION_LOCK, source_file_lock(settings.source_dir):
            source_name, source_hash, info, source_existed = await _persist_processing_upload(
                file, settings
            )
            try:
                expected = processing_output_size(info, outscale)
                settings.admit_upscale_processing(info.width, info.height)
                settings.admit_processing_output_dimensions(*expected)
            except BaseException:
                if not source_existed:
                    _remove_orphan_source(settings.source_dir, source_name, store)
                raise
            request: dict[str, object] = {
                "task_type": "upscale",
                "model": model,
                "outscale": outscale,
                "tile": tile,
                "source_image_name": source_name,
                "source_image_sha256": source_hash,
                "source_identity_sha256": source_hash,
                "requested_width": info.width,
                "requested_height": info.height,
                "expected_width": expected[0],
                "expected_height": expected[1],
                "expected_mode": "RGB",
            }
            try:
                task = store.admit(idempotency_key, request, "upscale")
            except IdempotencyConflict as exc:
                if not source_existed:
                    _remove_orphan_source(settings.source_dir, source_name, store)
                raise HTTPException(409, "idempotency key conflicts with another request") from exc
            except QueueFull as exc:
                if not source_existed:
                    _remove_orphan_source(settings.source_dir, source_name, store)
                raise HTTPException(503, "task queue is full") from exc
            except Exception:
                if not source_existed:
                    _remove_orphan_source(settings.source_dir, source_name, store)
                raise
            if task.status in {"succeeded", "failed"}:
                _remove_orphan_source(settings.source_dir, source_name, store)
        return _safe_processing_task(task)

    @app.post("/v1/background-removal-tasks", status_code=202)
    async def admit_background_removal_task(
        file: Annotated[UploadFile, File()],
        model: Annotated[
            Literal["bria-rmbg-2.0", "birefnet-hr-matting"],
            Query(),
        ],
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
        alpha_blur: Annotated[float, Query(ge=0, le=20)] = 0,
        alpha_erode: Annotated[int, Query(ge=0, le=100)] = 0,
        alpha_dilate: Annotated[int, Query(ge=0, le=100)] = 0,
        alpha_threshold: Annotated[int, Query(ge=0, le=255)] = 0,
        birefnet_inference_size: Annotated[int, Query(ge=512, le=4096)] = 2048,
        birefnet_foreground_refinement: bool = False,
        model_input_size: Annotated[int, Query(ge=512, le=2048)] = 1024,
        despill_enabled: bool = False,
        despill_color: Annotated[
            Literal["black", "white", "green", "blue", "custom"], Query()
        ] = "black",
        despill_hex_color: Annotated[str, Query(pattern="^[0-9A-Fa-f]{6}$")] = "000000",
    ) -> dict[str, object]:
        if not IDEMPOTENCY_PATTERN.fullmatch(idempotency_key):
            await file.close()
            raise HTTPException(422, "invalid idempotency key")
        with _ADMISSION_LOCK, source_file_lock(settings.source_dir):
            source_name, source_hash, info, source_existed = await _persist_processing_upload(
                file, settings
            )
            try:
                settings.admit_processing_output_dimensions(info.width, info.height)
            except BaseException:
                if not source_existed:
                    _remove_orphan_source(settings.source_dir, source_name, store)
                raise
            request = {
                "task_type": "background-removal",
                "model": model,
                "alpha_blur": alpha_blur,
                "alpha_erode": alpha_erode,
                "alpha_dilate": alpha_dilate,
                "alpha_threshold": alpha_threshold,
                "birefnet_inference_size": birefnet_inference_size,
                "birefnet_foreground_refinement": birefnet_foreground_refinement,
                "model_input_size": model_input_size,
                "despill_enabled": despill_enabled,
                "despill_color": despill_color,
                "despill_hex_color": despill_hex_color.lower(),
                "source_image_name": source_name,
                "source_image_sha256": source_hash,
                "source_identity_sha256": source_hash,
                "requested_width": info.width,
                "requested_height": info.height,
                "expected_width": info.width,
                "expected_height": info.height,
                "expected_mode": "RGBA",
            }
            try:
                task = store.admit(idempotency_key, request, "background-removal")
            except IdempotencyConflict as exc:
                if not source_existed:
                    _remove_orphan_source(settings.source_dir, source_name, store)
                raise HTTPException(409, "idempotency key conflicts with another request") from exc
            except QueueFull as exc:
                if not source_existed:
                    _remove_orphan_source(settings.source_dir, source_name, store)
                raise HTTPException(503, "task queue is full") from exc
            except Exception:
                if not source_existed:
                    _remove_orphan_source(settings.source_dir, source_name, store)
                raise
            if task.status in {"succeeded", "failed"}:
                _remove_orphan_source(settings.source_dir, source_name, store)
        return _safe_processing_task(task)

    def processing_task(task_id: str, capability: TaskKind) -> TaskRecord:
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise HTTPException(404, "task not found")
        try:
            task = store.get(task_id)
        except KeyError as exc:
            raise HTTPException(404, "task not found") from exc
        if task.task_kind != capability:
            raise HTTPException(404, "task not found")
        return task

    def processing_image(task_id: str, capability: TaskKind) -> FileResponse:
        task = processing_task(task_id, capability)
        if task.status != "succeeded" or task.image_name != f"{task_id}.png":
            raise HTTPException(409, "processing result is not available")
        assert task.image_name is not None
        path = settings.output_dir / task.image_name
        try:
            output_hash, info = validate_processing_output(path, task, settings)
            if (
                output_hash != task.output_sha256
                or info.width != task.output_width
                or info.height != task.output_height
                or info.mode != task.output_mode
            ):
                raise ValueError("processing output metadata mismatch")
        except Exception as exc:
            logger.error(
                "processing result validation failed: capability=%s task_id=%s",
                capability,
                task_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise HTTPException(503, "processing result is unavailable") from exc
        return FileResponse(path, media_type="image/png", filename=task.image_name)

    @app.get("/v1/upscale-tasks/{task_id}")
    def upscale_task_status(task_id: str) -> dict[str, object]:
        return _safe_processing_task(processing_task(task_id, "upscale"))

    @app.get("/v1/upscale-tasks/{task_id}/image")
    def upscale_task_image(task_id: str) -> FileResponse:
        return processing_image(task_id, "upscale")

    @app.get("/v1/background-removal-tasks/{task_id}")
    def background_task_status(task_id: str) -> dict[str, object]:
        return _safe_processing_task(processing_task(task_id, "background-removal"))

    @app.get("/v1/background-removal-tasks/{task_id}/image")
    def background_task_image(task_id: str) -> FileResponse:
        return processing_image(task_id, "background-removal")

    @app.post("/v1/upscale", response_class=Response)
    async def upscale(
        file: Annotated[UploadFile, File()],
        model: Annotated[Literal["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], Query()],
        outscale: Annotated[float, Query(ge=1, le=4)],
        tile: Annotated[int, Query(ge=0, le=1024)],
    ) -> Response:
        if tile != 0 and tile % 32:
            raise HTTPException(422, "tile must be zero or a multiple of 32")
        info = await _validated_sync_upload(file, settings)
        try:
            expected = processing_output_size(info, outscale)
            settings.admit_upscale_processing(info.width, info.height)
            settings.admit_processing_output_dimensions(*expected)
        except BaseException:
            await file.close()
            raise
        try:
            encoded = workers.upscale(file.file, model=model, outscale=outscale, tile=tile)
        finally:
            await file.close()
        try:
            validate_png_output(
                encoded,
                expected_size=expected,
                required_mode="RGB",
                max_bytes=settings.processing_max_encoded_output_bytes,
                max_pixels=settings.processing_max_output_pixels,
                max_decoded_bytes=settings.processing_max_decoded_output_bytes,
            )
        except Exception:
            if not isinstance(encoded, bytes):
                encoded.close()
            raise
        return _worker_image_response(encoded)

    @app.post("/v1/background-removal", response_class=Response)
    async def background_removal(
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
        info = await _validated_sync_upload(file, settings)
        try:
            settings.admit_processing_output_dimensions(info.width, info.height)
        except BaseException:
            await file.close()
            raise
        parameters = {
            "model": model,
            "alpha_blur": alpha_blur,
            "alpha_erode": alpha_erode,
            "alpha_dilate": alpha_dilate,
            "alpha_threshold": alpha_threshold,
            "birefnet_inference_size": birefnet_inference_size,
            "birefnet_foreground_refinement": birefnet_foreground_refinement,
            "model_input_size": model_input_size,
        }
        try:
            encoded = workers.background(file.file, **parameters)
        finally:
            await file.close()
        try:
            validate_png_output(
                encoded,
                expected_size=(info.width, info.height),
                required_mode="RGBA",
                max_bytes=settings.processing_max_encoded_output_bytes,
                max_pixels=settings.processing_max_output_pixels,
                max_decoded_bytes=settings.processing_max_decoded_output_bytes,
            )
        except Exception:
            if not isinstance(encoded, bytes):
                encoded.close()
            raise
        return _worker_image_response(encoded)

    @app.post("/v1/image-edits", status_code=202)
    async def admit_image_edit(
        file: Annotated[UploadFile, File()],
        model: Annotated[Literal["longcat-image-edit", "longcat-image-edit-turbo"], Form()],
        prompt: Annotated[str, Form(min_length=1, max_length=4000)],
        seed: Annotated[int, Form(ge=0, le=2**32 - 1)],
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
        negative_prompt: Annotated[str, Form(max_length=4000)] = "",
    ) -> dict[str, object]:
        if not IDEMPOTENCY_PATTERN.fullmatch(idempotency_key):
            raise HTTPException(422, "invalid idempotency key")
        raw_status = workers.health().get("generation", {})
        _, editing_status = _generation_health(settings, raw_status)
        editing_models = editing_status.get("models")
        selected_status = editing_models.get(model) if isinstance(editing_models, dict) else None
        if not isinstance(selected_status, dict) or not bool(selected_status.get("ready")):
            raise WorkerUnavailable("image-editing capability is unavailable")
        data = await _read_upload(file, settings.max_upload_bytes)
        info = validate_image(
            data,
            max_bytes=settings.max_upload_bytes,
            max_width=settings.max_input_width,
            max_height=settings.max_input_height,
            max_pixels=settings.max_input_pixels,
            max_decoded_bytes=settings.max_decoded_input_bytes,
        )
        original_hash = hashlib.sha256(data).hexdigest()
        normalized = _normalize_source(data)
        with _ADMISSION_LOCK, source_file_lock(settings.source_dir):
            candidate_name = f"{original_hash}-{hashlib.sha256(normalized).hexdigest()}.png"
            source_existed = (settings.source_dir / candidate_name).is_file()
            try:
                source_name = _persist_source_atomically(
                    settings.source_dir, normalized, original_hash
                )
            except OSError as exc:
                if not source_existed:
                    _remove_orphan_source(settings.source_dir, candidate_name, store)
                logger.error(
                    "image-edit source persistence failed: source_name=%s",
                    candidate_name,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                raise HTTPException(503, "source image could not be persisted") from exc
            request: dict[str, object] = {
                "task_type": "image-edit",
                "model": model,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "source_image_sha256": original_hash,
                "source_image_name": source_name,
                "source_width": info.width,
                "source_height": info.height,
            }
            try:
                task = store.admit(idempotency_key, request)
            except IdempotencyConflict as exc:
                if not source_existed:
                    _remove_orphan_source(settings.source_dir, source_name, store)
                raise HTTPException(409, "idempotency key conflicts with another request") from exc
            except QueueFull as exc:
                if not source_existed:
                    _remove_orphan_source(settings.source_dir, source_name, store)
                raise HTTPException(503, "generation queue is full") from exc
            except Exception:
                if not source_existed:
                    _remove_orphan_source(settings.source_dir, source_name, store)
                logger.exception("image-edit admission failed after source persistence")
                raise
            if task.status in {"succeeded", "failed"}:
                _remove_orphan_source(settings.source_dir, source_name, store)
        return _safe_task(task)

    @app.get("/v1/image-edits/{task_id}")
    def image_edit_status(task_id: str) -> dict[str, object]:
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise HTTPException(404, "task not found")
        try:
            task = store.get(task_id)
        except KeyError as exc:
            raise HTTPException(404, "task not found") from exc
        if task.task_kind != "generation" or task.request.get("task_type") != "image-edit":
            raise HTTPException(404, "task not found")
        return _safe_task(task)

    @app.get("/v1/image-edits/{task_id}/image")
    def image_edit_image(task_id: str) -> FileResponse:
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise HTTPException(404, "task not found")
        try:
            task = store.get(task_id)
        except KeyError as exc:
            raise HTTPException(404, "task not found") from exc
        if task.task_kind != "generation" or task.request.get("task_type") != "image-edit":
            raise HTTPException(404, "task not found")
        image_name = task.image_name
        if task.status != "succeeded" or image_name != f"{task_id}.png":
            raise HTTPException(409, "image-edit result is not available")
        assert image_name is not None
        path = settings.output_dir / image_name
        if not path.is_file():
            logger.error("image-edit result missing for succeeded task: task_id=%s", task_id)
            raise HTTPException(503, "image-edit result is unavailable")
        return FileResponse(path, media_type="image/png", filename=f"{task_id}.png")

    @app.post("/v1/generations", status_code=202)
    def admit_generation(
        body: GenerationRequest,
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
    ) -> dict[str, object]:
        if not IDEMPOTENCY_PATTERN.fullmatch(idempotency_key):
            raise HTTPException(422, "invalid idempotency key")
        if body.prompt is not None and settings.magic_prompt_backend is None:
            raise HTTPException(422, "plain prompt expansion is not configured")
        raw_status = workers.health().get("generation", {})
        generation_status, _ = _generation_health(settings, raw_status)
        if not bool(generation_status["ready"]):
            raise WorkerUnavailable("generation capability is unavailable")
        request = body.model_dump(exclude_none=True) | {"model": "ideogram-4-nf4"}
        try:
            task = store.admit(idempotency_key, request)
        except IdempotencyConflict as exc:
            raise HTTPException(409, "idempotency key conflicts with another request") from exc
        except QueueFull as exc:
            raise HTTPException(503, "generation queue is full") from exc
        return _safe_task(task)

    @app.get("/v1/generations/{task_id}")
    def generation_status(task_id: str) -> dict[str, object]:
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise HTTPException(404, "task not found")
        try:
            task = store.get(task_id)
        except KeyError as exc:
            raise HTTPException(404, "task not found") from exc
        if task.task_kind != "generation" or task.request.get("task_type") == "image-edit":
            raise HTTPException(404, "task not found")
        return _safe_task(task)

    @app.get("/v1/generations/{task_id}/image")
    def generation_image(task_id: str) -> FileResponse:
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise HTTPException(404, "task not found")
        try:
            task = store.get(task_id)
        except KeyError as exc:
            raise HTTPException(404, "task not found") from exc
        if task.task_kind != "generation" or task.request.get("task_type") == "image-edit":
            raise HTTPException(404, "task not found")
        if task.status != "succeeded" or task.image_name != f"{task_id}.png":
            raise HTTPException(409, "generation image is not available")
        path = settings.output_dir / task.image_name
        if not path.is_file():
            logger.error("generation image missing for succeeded task")
            raise HTTPException(503, "generation image is unavailable")
        return FileResponse(path, media_type="image/png", filename=f"{task_id}.png")

    return app
