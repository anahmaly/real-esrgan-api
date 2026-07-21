from __future__ import annotations

import os
import threading
import time
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI
from PIL import Image

from image_api.generation import GenerationRunner, recover_interrupted_tasks, start_worker_heartbeat
from image_api.lane import GpuLane
from image_api.store import TaskStore

app = FastAPI(title="image-api-test-generation-worker", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict[str, object]:
    return {"ready": True, "loaded": False, "device": "cpu-test", "weightsAvailable": True}


@app.post("/internal/unload")
def unload() -> dict[str, object]:
    return {"unloaded": True}


def fake_model(request: dict[str, object]) -> bytes:
    if request.get("task_type") == "image-edit":
        state = Path(os.getenv("IMAGE_API_STATE_DIR", "/state"))
        with Image.open(state / "sources" / str(request["source_image_name"])) as source:
            image = source.convert("RGB")
    else:
        width = request.get("width")
        height = request.get("height")
        if type(width) is not int or type(height) is not int:
            raise ValueError("invalid test generation dimensions")
        image = Image.new("RGB", (width, height), (20, 30, 40))
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def _serve() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9003, log_level="warning")


def main() -> None:
    state = Path(os.getenv("IMAGE_API_STATE_DIR", "/state"))
    start_worker_heartbeat(state / "generation-worker.heartbeat")
    threading.Thread(target=_serve, name="fake-generation-control", daemon=True).start()
    store = TaskStore(state / "tasks.sqlite3")
    recover_interrupted_tasks(store, state / "outputs")
    runner = GenerationRunner(
        store, GpuLane(state / "gpu-lane.lock", 30), state / "outputs", fake_model
    )
    while True:
        if not runner.run_one():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
