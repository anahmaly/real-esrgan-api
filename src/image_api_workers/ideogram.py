from __future__ import annotations

import gc
import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from image_api.config import ideogram_weights_available

logger = logging.getLogger(__name__)


class IdeogramRuntimeUnavailable(RuntimeError):
    pass


class IdeogramModel:
    """Local-mounted, offline adapter around the official Ideogram 4 pipeline."""

    def __init__(
        self,
        weights_path: Path,
        *,
        pipeline_factory: Callable[[], Any] | None = None,
        magic_prompt_factory: Callable[[str], Any] | None = None,
        cuda_available: Callable[[], bool] | None = None,
        status_path: Path | None = None,
    ) -> None:
        self.weights_path = weights_path
        self._pipeline_factory = pipeline_factory
        self._magic_prompt_factory = magic_prompt_factory
        self._cuda_available = cuda_available
        self._status_path = status_path
        self._pipeline: Any | None = None
        self._lock = threading.RLock()

    def _write_status(self, state: str) -> None:
        if self._status_path is None:
            return
        self._status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._status_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps({"state": state, "loaded": state == "loaded"}))
        os.replace(temporary, self._status_path)

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        self._write_status("loading")
        if not self.weights_path.is_dir():
            self._write_status("unloaded")
            raise IdeogramRuntimeUnavailable("configured Ideogram weight mount is unavailable")
        repository = os.getenv("IMAGE_API_IDEOGRAM_REPOSITORY_ID", "ideogram-ai/ideogram-4-nf4")
        if self._pipeline_factory is None and not ideogram_weights_available(
            self.weights_path, repository
        ):
            self._write_status("unloaded")
            raise IdeogramRuntimeUnavailable("configured Ideogram weight mount is incomplete")
        cuda = self._cuda_available
        if cuda is None:
            import torch

            cuda = torch.cuda.is_available
        if not cuda():
            self._write_status("unloaded")
            raise IdeogramRuntimeUnavailable("Ideogram 4 NF4 requires CUDA")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HOME"] = str(self.weights_path)
        if self._pipeline_factory is not None:
            self._pipeline = self._pipeline_factory()
            self._write_status("loaded")
            return self._pipeline
        try:
            import torch
            from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig

            self._pipeline = Ideogram4Pipeline.from_pretrained(
                config=Ideogram4PipelineConfig(weights_repo=repository),
                device="cuda",
                dtype=torch.bfloat16,
            )
        except Exception as exc:
            self._write_status("unloaded")
            logger.error(
                "Ideogram runtime initialization failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise IdeogramRuntimeUnavailable("Ideogram runtime initialization failed") from None
        self._write_status("loaded")
        return self._pipeline

    def _run(self, request: dict[str, object]) -> bytes:
        width = request.get("width")
        height = request.get("height")
        seed = request.get("seed")
        preset_name = request.get("sampler_preset")
        if (
            type(width) is not int
            or type(height) is not int
            or not 256 <= width <= 2048
            or not 256 <= height <= 2048
            or width % 16
            or height % 16
            or type(seed) is not int
            or not 0 <= seed <= 2**32 - 1
            or not isinstance(preset_name, str)
            or preset_name not in {"V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12"}
        ):
            raise ValueError("invalid persisted generation request")
        caption = request.get("structured_caption")
        plain_prompt = request.get("prompt")
        if (caption is None) == (plain_prompt is None):
            raise ValueError("generation request must have exactly one caption mode")
        if caption is not None and not isinstance(caption, dict):
            raise ValueError("structured caption must be an object")
        if plain_prompt is not None and not isinstance(plain_prompt, str):
            raise ValueError("plain prompt must be a string")
        pipeline = self._load()
        if caption is not None:
            prompt = json.dumps(caption, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        else:
            backend = os.getenv("IMAGE_API_MAGIC_PROMPT_BACKEND")
            key = os.getenv("IMAGE_API_MAGIC_PROMPT_API_KEY")
            if not backend or not key:
                raise IdeogramRuntimeUnavailable("configured magic prompt backend is unavailable")
            if self._magic_prompt_factory is not None:
                expander = self._magic_prompt_factory(backend)
            else:
                from ideogram4 import MAGIC_PROMPTS

                if backend not in MAGIC_PROMPTS:
                    raise IdeogramRuntimeUnavailable(
                        "configured magic prompt backend is unsupported"
                    )
                expander = MAGIC_PROMPTS[backend](api_key=key)
            from ideogram4 import aspect_ratio_from_size

            try:
                prompt = expander.expand(
                    plain_prompt,
                    aspect_ratio=aspect_ratio_from_size(width, height),
                )
            except Exception as exc:
                logger.error(
                    "Ideogram magic prompt expansion failed: backend=%s",
                    backend,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                raise IdeogramRuntimeUnavailable("magic prompt expansion failed") from None
            if not prompt:
                raise IdeogramRuntimeUnavailable("magic prompt expansion failed")
        from ideogram4 import PRESETS

        preset = PRESETS[preset_name]
        try:
            images = pipeline(
                prompt,
                height=height,
                width=width,
                num_steps=preset.num_steps,
                guidance_schedule=preset.guidance_schedule,
                mu=preset.mu,
                std=preset.std,
                seed=seed,
                raise_on_caption_issues=True,
            )
        except Exception as exc:
            logger.error(
                "Ideogram inference failed: preset=%s dimensions=%sx%s",
                preset_name,
                width,
                height,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise IdeogramRuntimeUnavailable("Ideogram inference failed") from None
        from io import BytesIO

        output = BytesIO()
        images[0].save(output, "PNG")
        return output.getvalue()

    def __call__(self, request: dict[str, object]) -> bytes:
        with self._lock:
            return self._run(request)

    def unload(self) -> None:
        with self._lock:
            pipeline = self._pipeline
            self._pipeline = None
            hook_error: Exception | None = None
            if pipeline is not None:
                for hook_name in ("remove_all_hooks", "_remove_all_hooks"):
                    remove_hooks = getattr(pipeline, hook_name, None)
                    if callable(remove_hooks):
                        try:
                            remove_hooks()
                        except Exception as exc:
                            hook_error = exc
                            logger.error(
                                "Ideogram offload hook removal failed",
                                exc_info=(type(exc), exc, exc.__traceback__),
                            )
                        break
                del pipeline
            gc.collect()
            cuda_error: Exception | None = None
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
                cuda_error = exc
                logger.error(
                    "Ideogram CUDA cache release failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
            self._write_status("unloaded")
            if cuda_error is not None:
                raise RuntimeError("Ideogram CUDA cache release failed") from cuda_error
            if hook_error is not None:
                raise RuntimeError("Ideogram offload hook removal failed") from hook_error

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None
