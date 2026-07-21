from __future__ import annotations

import atexit
import logging
import os
import threading
import time

from fastapi import FastAPI, HTTPException

from image_api.config import Settings, ideogram_weights_available, longcat_weights_available
from image_api.generation import GenerationRunner, recover_interrupted_tasks, start_worker_heartbeat
from image_api.lane import GpuLane
from image_api.store import TaskStore
from image_api.workers import PeerEvictor
from image_api_workers.generation_models import GenerationModels, LongCatImageEditModel
from image_api_workers.ideogram import IdeogramModel

logging.basicConfig(level=os.getenv("IMAGE_API_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def create_control_app(models: GenerationModels, settings: Settings) -> FastAPI:
    app = FastAPI(title="image-api-generation-worker", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> dict[str, object]:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except (ImportError, AttributeError, RuntimeError):
            cuda_available = False
        repository = os.getenv("IMAGE_API_IDEOGRAM_REPOSITORY_ID", "ideogram-ai/ideogram-4-nf4")
        mounted = {
            "ideogram-4-nf4": ideogram_weights_available(
                settings.ideogram_weights_path, repository
            ),
            "longcat-image-edit": longcat_weights_available(
                settings.longcat_edit_weights_path, settings.longcat_edit_revision
            ),
            "longcat-image-edit-turbo": longcat_weights_available(
                settings.longcat_edit_turbo_weights_path,
                settings.longcat_edit_turbo_revision,
            ),
        }
        result: dict[str, object] = {
            "ready": cuda_available and any(mounted.values()),
            "loaded": models.loaded_model is not None,
            "device": "cuda" if cuda_available else "unavailable",
            "weightsAvailable": all(mounted.values()),
            "models": {
                name: {"weightsAvailable": available, "loaded": models.loaded_model == name}
                for name, available in mounted.items()
            },
        }
        if models.loaded_model is not None:
            result["loadedModel"] = models.loaded_model
        return result

    @app.post("/internal/unload")
    def unload() -> dict[str, object]:
        try:
            models.unload()
        except Exception as exc:
            logger.error(
                "generation worker unload failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise HTTPException(500, "model unload failed") from exc
        return {"unloaded": True, "loaded": False}

    return app


def _serve_control(app: FastAPI) -> None:
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("IMAGE_API_GENERATION_WORKER_PORT", "9003")),
        log_level=os.getenv("IMAGE_API_LOG_LEVEL", "info").lower(),
    )


def main() -> None:
    settings = Settings.from_env()
    state = settings.state_dir
    start_worker_heartbeat(settings.generation_heartbeat_path)
    store = TaskStore(settings.database_path, settings.max_queue_depth)
    recovered = recover_interrupted_tasks(store, settings.output_dir)
    if recovered:
        logger.warning("Reconciled interrupted generation tasks: count=%s", recovered)
    models = GenerationModels(
        IdeogramModel(settings.ideogram_weights_path),
        LongCatImageEditModel(
            {
                "longcat-image-edit": settings.longcat_edit_weights_path,
                "longcat-image-edit-turbo": settings.longcat_edit_turbo_weights_path,
            },
            settings.source_dir,
            revisions={
                "longcat-image-edit": settings.longcat_edit_revision,
                "longcat-image-edit-turbo": settings.longcat_edit_turbo_revision,
            },
        ),
        status_path=state / "generation-model-status.json",
    )

    def shutdown_unload() -> None:
        try:
            models.unload()
        except Exception as exc:
            logger.error(
                "generation worker shutdown unload failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    atexit.register(shutdown_unload)
    control = threading.Thread(
        target=_serve_control,
        args=(create_control_app(models, settings),),
        name="generation-control",
        daemon=True,
    )
    control.start()
    runner = GenerationRunner(
        store,
        GpuLane(settings.gpu_lane_path, settings.lane_timeout_seconds),
        settings.output_dir,
        models,
        peer_evictor=PeerEvictor(
            (
                os.getenv("IMAGE_API_UPSCALE_WORKER_URL", "http://upscale-worker:9001"),
                os.getenv("IMAGE_API_BACKGROUND_WORKER_URL", "http://background-worker:9002"),
            )
        ),
    )
    poll = float(os.getenv("IMAGE_API_GENERATION_POLL_SECONDS", "0.5"))
    while True:
        if not runner.run_one():
            time.sleep(poll)


if __name__ == "__main__":
    main()
