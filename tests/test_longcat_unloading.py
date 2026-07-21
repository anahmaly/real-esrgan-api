from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from helpers import png
from image_api.app import create_app
from image_api.config import Settings, longcat_weights_available
from image_api.generation import GenerationRunner
from image_api.lane import GpuLane
from image_api.store import TaskStore
from image_api.workers import FakeWorkerClient, HttpWorkerClient, WorkerUnavailable
from image_api_workers.generation_models import GenerationModels, LongCatImageEditModel


class EditPipeline:
    def __init__(self, name: str, events: list[object]) -> None:
        self.name = name
        self.events = events
        self.hooks_removed = 0

    def enable_model_cpu_offload(self) -> None:
        self.events.append(("offload", self.name))

    def remove_all_hooks(self) -> None:
        self.hooks_removed += 1
        self.events.append(("hooks", self.name))

    def __call__(self, image: Image.Image, prompt: str, **kwargs: object):
        self.events.append(("call", self.name, image.copy(), prompt, kwargs))
        return SimpleNamespace(images=[Image.new("RGB", (1024, 551))])


class FakeIdeogram:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.loaded = False

    def __call__(self, request: dict[str, object]) -> bytes:
        self.loaded = True
        self.events.append(("ideogram", request["seed"]))
        return png("RGB", (256, 256))

    def unload(self) -> None:
        self.events.append("unload-ideogram")
        self.loaded = False


def test_standard_and_turbo_dispatch_actual_image_with_official_defaults(tmp_path) -> None:
    events: list[object] = []
    pipelines: dict[str, EditPipeline] = {}
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "source.png").write_bytes(png("RGB", (13, 7)))

    def factory(model: str, _path: Path) -> EditPipeline:
        pipeline = EditPipeline(model, events)
        pipelines[model] = pipeline
        return pipeline

    adapter = LongCatImageEditModel(
        {
            "longcat-image-edit": tmp_path,
            "longcat-image-edit-turbo": tmp_path,
        },
        source_dir,
        pipeline_factory=factory,
        cuda_available=lambda: True,
    )
    base = {
        "task_type": "image-edit",
        "source_image_name": "source.png",
        "prompt": "make it blue",
        "negative_prompt": "",
        "seed": 43,
    }
    adapter(base | {"model": "longcat-image-edit"})
    adapter(base | {"model": "longcat-image-edit-turbo"})

    standard = next(
        event
        for event in events
        if isinstance(event, tuple) and event[:2] == ("call", "longcat-image-edit")
    )
    turbo = next(
        event
        for event in events
        if isinstance(event, tuple) and event[:2] == ("call", "longcat-image-edit-turbo")
    )
    assert standard[2].size == turbo[2].size == (13, 7)
    assert standard[2].mode == turbo[2].mode == "RGB"
    assert standard[4] == {
        "negative_prompt": "",
        "guidance_scale": 4.5,
        "num_inference_steps": 50,
        "num_images_per_prompt": 1,
        "generator": ("cpu", 43),
    }
    assert turbo[4] == standard[4] | {"guidance_scale": 1.0, "num_inference_steps": 8}
    assert pipelines["longcat-image-edit"].hooks_removed == 1


def test_generation_model_switching_unloads_longcat_and_preserves_ideogram(tmp_path) -> None:
    events: list[object] = []
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "source.png").write_bytes(png())
    ideogram = FakeIdeogram(events)
    longcat = LongCatImageEditModel(
        {"longcat-image-edit": tmp_path, "longcat-image-edit-turbo": tmp_path},
        source_dir,
        pipeline_factory=lambda model, _path: EditPipeline(model, events),
        cuda_available=lambda: True,
    )
    models = GenerationModels(ideogram, longcat)
    models(
        {
            "model": "longcat-image-edit",
            "source_image_name": "source.png",
            "prompt": "x",
            "negative_prompt": "",
            "seed": 1,
        }
    )
    models(
        {
            "model": "longcat-image-edit-turbo",
            "source_image_name": "source.png",
            "prompt": "x",
            "negative_prompt": "",
            "seed": 1,
        }
    )
    models({"model": "ideogram-4-nf4", "width": 256, "height": 256, "seed": 1})
    assert events[-2:] == [("hooks", "longcat-image-edit-turbo"), ("ideogram", 1)]
    assert models.loaded_model == "ideogram-4-nf4"


def test_cross_process_peer_eviction_is_inside_lane_and_before_model_load(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    store.admit("peer-order", {"width": 256, "height": 256, "seed": 1})
    lane = GpuLane(tmp_path / "gpu.lock")
    events: list[str] = []

    def evict() -> None:
        assert lane.status()["activeCapability"] == "generation"
        events.append("evict")

    def model(_request: dict[str, object]) -> bytes:
        events.append("model")
        return png("RGB", (256, 256))

    assert GenerationRunner(store, lane, tmp_path / "outputs", model, peer_evictor=evict).run_one()
    assert events == ["evict", "model"]


def test_peer_eviction_failure_fails_closed_before_inference(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.admit("peer-fail", {"width": 256, "height": 256, "seed": 1})
    called = False

    def model(_request: dict[str, object]) -> bytes:
        nonlocal called
        called = True
        return png("RGB", (256, 256))

    def fail() -> None:
        raise WorkerUnavailable("private peer detail")

    runner = GenerationRunner(
        store, GpuLane(tmp_path / "gpu.lock"), tmp_path / "out", model, peer_evictor=fail
    )
    assert runner.run_one()
    assert called is False
    assert store.get(task.task_id).status == "failed"


def test_explicit_unload_is_lane_serialized_and_clears_bounded_health(tmp_path) -> None:
    settings = Settings.for_tests(tmp_path, lane_timeout_seconds=0.05)
    workers = FakeWorkerClient()
    workers.set_loaded("generation", "longcat-image-edit")
    client = TestClient(
        create_app(settings=settings, store=TaskStore(settings.database_path), workers=workers)
    )
    entered = threading.Event()
    release = threading.Event()

    def active_inference() -> None:
        with GpuLane(settings.gpu_lane_path, 1).acquire("generation"):
            entered.set()
            release.wait(2)

    thread = threading.Thread(target=active_inference)
    thread.start()
    assert entered.wait(2)
    busy = client.post("/v1/models/unload")
    assert busy.status_code == 503
    assert workers.unload_calls == 0
    release.set()
    thread.join(2)

    response = client.post("/v1/models/unload")
    assert response.status_code == 200
    assert set(response.json()["workers"]) == {"upscale", "background-removal", "generation"}
    assert workers.unload_calls == 1
    generation = client.get("/health").json()["capabilities"]["generation"]
    assert generation["loaded"] is False
    assert "loadedModel" not in generation


def test_unload_worker_failure_is_bounded_per_worker(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "background-worker":
            return httpx.Response(500, text="private traceback /models/secret")
        return httpx.Response(200, json={"unloaded": True})

    workers = HttpWorkerClient(
        "http://upscale-worker",
        "http://background-worker",
        timeout_seconds=1,
        max_output_bytes=1000,
        transport=httpx.MockTransport(handler),
        generation_url="http://generation-worker",
    )
    settings = Settings.for_tests(tmp_path, lane_timeout_seconds=0.05)
    client = TestClient(
        create_app(settings=settings, store=TaskStore(settings.database_path), workers=workers)
    )
    response = client.post("/v1/models/unload")
    assert response.status_code == 503
    assert response.json()["workers"] == {
        "upscale": {"unloaded": True},
        "background-removal": {"unloaded": False, "error": "worker_unavailable"},
        "generation": {"unloaded": True},
    }
    assert "private traceback" not in response.text
    assert "/models/" not in response.text


def _snapshot(root: Path, revision: str) -> Path:
    root.mkdir(parents=True)
    required = [
        "config.json",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/generation_config.json",
        "text_encoder/preprocessor_config.json",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/model-00001-of-00001.safetensors",
        "text_processor/chat_template.json",
        "text_processor/config.json",
        "text_processor/merges.txt",
        "text_processor/preprocessor_config.json",
        "text_processor/special_tokens_map.json",
        "text_processor/tokenizer.json",
        "text_processor/tokenizer_config.json",
        "text_processor/vocab.json",
        "tokenizer/chat_template.json",
        "tokenizer/config.json",
        "tokenizer/merges.txt",
        "tokenizer/preprocessor_config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    for relative in required:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (root / "text_encoder/model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00001.safetensors"}})
    )
    (root / ".image-api-revision").write_text(revision)
    return root


def test_longcat_snapshot_validation_requires_exact_revision_and_real_shards(tmp_path) -> None:
    revision = "7b54ef423aa7854be7861600024be5c56ab7875a"
    snapshot = _snapshot(tmp_path / "standard", revision)
    assert longcat_weights_available(snapshot, revision) is True
    (snapshot / "text_encoder/model-00001-of-00001.safetensors").unlink()
    assert longcat_weights_available(snapshot, revision) is False
    assert longcat_weights_available(snapshot, "6a7262de5549f0bf0ec54c08ef7d283ef41f3214") is False


def test_cuda_disposal_is_conditional(monkeypatch, tmp_path) -> None:
    events: list[object] = []
    source = tmp_path / "sources"
    source.mkdir()
    (source / "source.png").write_bytes(png())
    adapter = LongCatImageEditModel(
        {"longcat-image-edit": tmp_path},
        source,
        pipeline_factory=lambda model, _path: EditPipeline(model, events),
        cuda_available=lambda: True,
    )
    adapter(
        {
            "model": "longcat-image-edit",
            "source_image_name": "source.png",
            "prompt": "x",
            "negative_prompt": "",
            "seed": 1,
        }
    )
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: (_ for _ in ()).throw(AssertionError("must not call")),
    )  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    adapter.unload()
    assert adapter.loaded_model is None
