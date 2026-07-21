from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from image_api.images import validate_png_output
from image_api.lane import GpuLane
from image_api.store import TaskStore

logger = logging.getLogger(__name__)
GenerationModel = Callable[[dict[str, object]], bytes]
FailureInjector = Callable[[str], None]
PeerEvictor = Callable[[], None]
GENERATION_MAX_ENCODED_BYTES = 100_000_000
GENERATION_MAX_OUTPUT_PIXELS = 80_000_000
GENERATION_OUTPUT_MODE = "RGB"


def _expected_output(request: dict[str, object]) -> tuple[int, int] | None:
    if request.get("task_type") == "image-edit":
        return None
    width = request.get("width")
    height = request.get("height")
    if type(width) is not int or type(height) is not int:
        raise ValueError("invalid persisted generation dimensions")
    return (width, height)


def write_worker_heartbeat(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def start_worker_heartbeat(path: Path, *, interval_seconds: float = 2.0) -> None:
    def heartbeat_loop() -> None:
        while True:
            try:
                write_worker_heartbeat(path)
            except OSError:
                safe_error = RuntimeError("state storage unavailable")
                logger.error(
                    "generation worker heartbeat failed",
                    exc_info=(type(safe_error), safe_error, safe_error.__traceback__),
                )
            time.sleep(interval_seconds)

    write_worker_heartbeat(path)
    threading.Thread(target=heartbeat_loop, name="generation-heartbeat", daemon=True).start()


def worker_heartbeat_alive(path: Path, *, max_age_seconds: float) -> bool:
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return 0 <= age <= max_age_seconds


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _clean_task_files(output_dir: Path, task_id: str, *, remove_final: bool) -> None:
    changed = False
    for temporary in output_dir.glob(f".{task_id}.*.tmp"):
        temporary.unlink(missing_ok=True)
        changed = True
    if remove_final:
        final_path = output_dir / f"{task_id}.png"
        if final_path.exists():
            final_path.unlink()
            changed = True
    if changed:
        _fsync_directory(output_dir)


def reconcile_task_output(store: TaskStore, task_id: str, output_dir: Path) -> bool:
    """Resolve one ambiguous running/publication transition without invoking the model."""
    task = store.get(task_id)
    image_name = f"{task_id}.png"
    if task.status == "succeeded" and task.image_name == image_name:
        _clean_task_files(output_dir, task_id, remove_final=False)
        return True
    valid = False
    final_path = output_dir / image_name
    try:
        with final_path.open("rb") as handle:
            encoded = handle.read(GENERATION_MAX_ENCODED_BYTES + 1)
        expected = _expected_output(task.request)
        validate_png_output(
            encoded,
            expected_size=expected,
            required_mode=GENERATION_OUTPUT_MODE,
            max_bytes=GENERATION_MAX_ENCODED_BYTES,
            max_pixels=(
                expected[0] * expected[1] if expected is not None else GENERATION_MAX_OUTPUT_PIXELS
            ),
        )
        valid = True
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        valid = False
    if valid and store.reconcile_success(task_id, image_name):
        _clean_task_files(output_dir, task_id, remove_final=False)
        return True
    _clean_task_files(output_dir, task_id, remove_final=True)
    if store.get(task_id).status == "running":
        store.fail(task_id, "worker_interrupted")
    return False


def recover_interrupted_tasks(store: TaskStore, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    running = store.running()
    for task in running:
        reconcile_task_output(store, task.task_id, output_dir)
    return len(running)


class GenerationRunner:
    def __init__(
        self,
        store: TaskStore,
        lane: GpuLane,
        output_dir: Path,
        model: GenerationModel,
        worker_id: str | None = None,
        failure_injector: FailureInjector | None = None,
        peer_evictor: PeerEvictor | None = None,
    ) -> None:
        self.store = store
        self.lane = lane
        self.output_dir = output_dir
        self.model = model
        self.worker_id = worker_id or f"generation-{uuid.uuid4().hex[:12]}"
        self.failure_injector = failure_injector
        self.peer_evictor = peer_evictor
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_one(self) -> bool:
        task = self.store.claim_next(self.worker_id)
        if task is None:
            return False
        try:
            with self.lane.acquire("generation"):
                if self.peer_evictor is not None:
                    self.peer_evictor()
                encoded = self.model(task.request)
            expected = _expected_output(task.request)
            validate_png_output(
                encoded,
                expected_size=expected,
                required_mode=GENERATION_OUTPUT_MODE,
                max_bytes=GENERATION_MAX_ENCODED_BYTES,
                max_pixels=(
                    expected[0] * expected[1]
                    if expected is not None
                    else GENERATION_MAX_OUTPUT_PIXELS
                ),
            )
            image_name = f"{task.task_id}.png"
            final_path = self.output_dir / image_name
            temporary = self.output_dir / f".{task.task_id}.{uuid.uuid4().hex}.tmp"
            try:
                with temporary.open("xb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                if self.failure_injector is not None:
                    self.failure_injector("after_file_fsync")
                os.replace(temporary, final_path)
                _fsync_directory(self.output_dir)
                if self.failure_injector is not None:
                    self.failure_injector("after_directory_fsync")
            finally:
                temporary.unlink(missing_ok=True)
            self.store.succeed(task.task_id, image_name)
        except Exception:
            logger.exception("generation task failed")
            try:
                reconcile_task_output(self.store, task.task_id, self.output_dir)
            except Exception:
                logger.exception("generation task reconciliation could not be persisted")
        return True
