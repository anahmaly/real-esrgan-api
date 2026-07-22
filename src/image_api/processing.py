from __future__ import annotations

import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import BinaryIO, Callable, Literal

from image_api.config import Settings
from image_api.generation import (
    SOURCE_NAME_PATTERN,
    _clean_task_files,
    _cleanup_task_source,
    _fsync_directory,
)
from image_api.images import ImageInfo, validate_image, validate_png_output
from image_api.lane import GpuLane
from image_api.store import TaskRecord, TaskStore

logger = logging.getLogger(__name__)
ProcessingCapability = Literal["upscale", "background-removal"]
ProcessingModel = Callable[[Path, dict[str, object]], bytes | BinaryIO]
FailureInjector = Callable[[str], None]
PeerEvictor = Callable[[], None]
_CHUNK_BYTES = 64 * 1024


def _processing_contract(task: TaskRecord) -> tuple[tuple[int, int], str]:
    if task.task_kind not in {"upscale", "background-removal"}:
        raise ValueError("task is not a processing task")
    width = task.request.get("expected_width")
    height = task.request.get("expected_height")
    mode = task.request.get("expected_mode")
    if type(width) is not int or type(height) is not int or mode not in {"RGB", "RGBA"}:
        raise ValueError("invalid persisted processing contract")
    required_mode = "RGB" if task.task_kind == "upscale" else "RGBA"
    if mode != required_mode:
        raise ValueError("persisted processing mode does not match capability")
    return (width, height), mode


def _sha256_file(path: Path, max_bytes: int) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("persisted file exceeds configured limit")
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source(task: TaskRecord, source_dir: Path, settings: Settings) -> Path:
    source_name = task.request.get("source_image_name")
    expected_hash = task.request.get("source_image_sha256")
    if (
        not isinstance(source_name, str)
        or SOURCE_NAME_PATTERN.fullmatch(source_name) is None
        or not isinstance(expected_hash, str)
        or len(expected_hash) != 64
    ):
        raise ValueError("invalid persisted processing source identity")
    source = source_dir / source_name
    actual_hash = _sha256_file(source, settings.processing_max_upload_bytes)
    if actual_hash != expected_hash:
        raise ValueError("persisted processing source hash mismatch")
    with source.open("rb") as handle:
        info = validate_image(
            handle,
            max_bytes=settings.processing_max_upload_bytes,
            max_width=settings.processing_max_input_width,
            max_height=settings.processing_max_input_height,
            max_pixels=settings.processing_max_input_pixels,
            max_decoded_bytes=settings.processing_max_decoded_input_bytes,
        )
    requested = (task.request.get("requested_width"), task.request.get("requested_height"))
    if requested != (info.width, info.height):
        raise ValueError("persisted processing source dimensions mismatch")
    return source


def validate_processing_output(
    path: Path, task: TaskRecord, settings: Settings
) -> tuple[str, ImageInfo]:
    expected, required_mode = _processing_contract(task)
    with path.open("rb") as handle:
        validate_png_output(
            handle,
            expected_size=expected,
            required_mode=required_mode,
            max_bytes=settings.processing_max_encoded_output_bytes,
            max_pixels=settings.processing_max_output_pixels,
            max_decoded_bytes=settings.processing_max_decoded_output_bytes,
        )
        info = validate_image(
            handle,
            max_bytes=settings.processing_max_encoded_output_bytes,
            max_width=expected[0],
            max_height=expected[1],
            max_pixels=settings.processing_max_output_pixels,
            max_decoded_bytes=settings.processing_max_decoded_output_bytes,
            worker_output=True,
        )
    return _sha256_file(path, settings.processing_max_encoded_output_bytes), info


def reconcile_processing_task(
    capability: ProcessingCapability,
    store: TaskStore,
    task_id: str,
    output_dir: Path,
    source_dir: Path,
    settings: Settings,
) -> bool:
    task = store.get(task_id)
    if task.task_kind != capability:
        raise ValueError("processing worker cannot reconcile another capability")
    image_name = f"{task_id}.png"
    final_path = output_dir / image_name
    try:
        output_hash, info = validate_processing_output(final_path, task, settings)
    except Exception:
        _clean_task_files(output_dir, task_id, remove_final=True)
        if store.get(task_id).status == "running":
            store.fail(task_id, "worker_interrupted")
        _cleanup_task_source(store, task_id, source_dir)
        return False
    reconciled = store.reconcile_success(
        task_id,
        image_name,
        output_sha256=output_hash,
        output_width=info.width,
        output_height=info.height,
        output_mode=info.mode,
    )
    if reconciled:
        _clean_task_files(output_dir, task_id, remove_final=False)
        _cleanup_task_source(store, task_id, source_dir)
    return reconciled


def recover_processing_tasks(
    capability: ProcessingCapability,
    store: TaskStore,
    output_dir: Path,
    source_dir: Path,
    settings: Settings,
) -> int:
    if capability not in {"upscale", "background-removal"}:
        raise ValueError("invalid processing capability")
    output_dir.mkdir(parents=True, exist_ok=True)
    running = store.running(capability)
    for task in running:
        reconcile_processing_task(capability, store, task.task_id, output_dir, source_dir, settings)
    return len(running)


class ProcessingRunner:
    def __init__(
        self,
        capability: ProcessingCapability,
        store: TaskStore,
        lane: GpuLane,
        source_dir: Path,
        output_dir: Path,
        model: ProcessingModel,
        settings: Settings,
        *,
        worker_id: str | None = None,
        peer_evictor: PeerEvictor | None = None,
        failure_injector: FailureInjector | None = None,
    ) -> None:
        if capability not in {"upscale", "background-removal"}:
            raise ValueError("invalid processing capability")
        self.capability: ProcessingCapability = capability
        self.store = store
        self.lane = lane
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.model = model
        self.settings = settings
        self.worker_id = worker_id or f"{capability}-{uuid.uuid4().hex[:12]}"
        self.peer_evictor = peer_evictor
        self.failure_injector = failure_injector
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write_output(
        self, task: TaskRecord, encoded: bytes | BinaryIO
    ) -> tuple[Path, str, ImageInfo]:
        temporary = self.output_dir / f".{task.task_id}.{uuid.uuid4().hex}.tmp"
        final_path = self.output_dir / f"{task.task_id}.png"
        source: BinaryIO | None = None
        try:
            with temporary.open("xb") as target:
                if isinstance(encoded, bytes):
                    if len(encoded) > self.settings.processing_max_encoded_output_bytes:
                        raise ValueError("processing output exceeds configured limit")
                    target.write(encoded)
                else:
                    source = encoded
                    source.seek(0)
                    total = 0
                    while chunk := source.read(_CHUNK_BYTES):
                        total += len(chunk)
                        if total > self.settings.processing_max_encoded_output_bytes:
                            raise ValueError("processing output exceeds configured limit")
                        target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            if self.failure_injector is not None:
                self.failure_injector("after_file_fsync")
            output_hash, info = validate_processing_output(temporary, task, self.settings)
            os.replace(temporary, final_path)
            _fsync_directory(self.output_dir)
            if self.failure_injector is not None:
                self.failure_injector("after_directory_fsync")
            return final_path, output_hash, info
        finally:
            temporary.unlink(missing_ok=True)
            if source is not None:
                source.close()

    def run_one(self) -> bool:
        task = self.store.claim_next(self.worker_id, self.capability)
        if task is None:
            return False
        try:
            source = _validate_source(task, self.source_dir, self.settings)
            with self.lane.acquire(self.capability):
                if self.peer_evictor is not None:
                    self.peer_evictor()
                encoded = self.model(source, task.request)
            final_path, output_hash, info = self._write_output(task, encoded)
            self.store.succeed(
                task.task_id,
                final_path.name,
                output_sha256=output_hash,
                output_width=info.width,
                output_height=info.height,
                output_mode=info.mode,
            )
            _cleanup_task_source(self.store, task.task_id, self.source_dir)
        except Exception as exc:
            logger.error(
                "processing task failed: capability=%s task_id=%s",
                self.capability,
                task.task_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            try:
                reconcile_processing_task(
                    self.capability,
                    self.store,
                    task.task_id,
                    self.output_dir,
                    self.source_dir,
                    self.settings,
                )
            except Exception:
                logger.exception("processing task reconciliation could not be persisted")
        return True
