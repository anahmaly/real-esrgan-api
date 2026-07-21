from __future__ import annotations

import logging
from io import BytesIO
from typing import Protocol

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


class WorkerUnavailable(RuntimeError):
    pass


class WorkerClient(Protocol):
    model_invocations: int
    model_loads: int

    def health(self) -> dict[str, dict[str, object]]: ...
    def upscale(self, data: bytes, **parameters: object) -> bytes: ...
    def background(self, data: bytes, **parameters: object) -> bytes: ...
    def unload_all(self) -> dict[str, dict[str, object]]: ...


class HttpWorkerClient:
    model_invocations = 0
    model_loads = 0

    def __init__(
        self,
        upscale_url: str,
        background_url: str,
        timeout_seconds: float,
        max_output_bytes: int,
        transport: httpx.BaseTransport | None = None,
        generation_url: str = "http://generation-worker:9003",
    ) -> None:
        self.urls = {
            "upscale": upscale_url.rstrip("/"),
            "background-removal": background_url.rstrip("/"),
            "generation": generation_url.rstrip("/"),
        }
        self.upscale_url = self.urls["upscale"]
        self.background_url = self.urls["background-removal"]
        self.max_output_bytes = max_output_bytes
        self.client = httpx.Client(timeout=httpx.Timeout(timeout_seconds), transport=transport)

    def _get_health(self, base: str) -> dict[str, object]:
        try:
            response = self.client.get(f"{base}/health", timeout=0.25)
            response.raise_for_status()
            body = response.json()
            raw_device = body.get("device")
            device = (
                raw_device if raw_device in {"cuda", "cpu-test", "unavailable"} else "unavailable"
            )
            result: dict[str, object] = {
                "ready": bool(body.get("ready")),
                "loaded": bool(body.get("loaded")),
                "device": device,
                "workerReachable": True,
            }
            loaded_model = body.get("loadedModel")
            if isinstance(loaded_model, str):
                result["loadedModel"] = loaded_model
            if "weightsAvailable" in body:
                result["weightsAvailable"] = bool(body["weightsAvailable"])
            models = body.get("models")
            if isinstance(models, dict):
                allowed_models = {
                    "ideogram-4-nf4",
                    "longcat-image-edit",
                    "longcat-image-edit-turbo",
                }
                result["models"] = {
                    name: {
                        "weightsAvailable": bool(value.get("weightsAvailable", False)),
                        "loaded": bool(value.get("loaded", False)),
                    }
                    for name, value in models.items()
                    if name in allowed_models and isinstance(value, dict)
                }
            return result
        except Exception as exc:
            logger.warning(
                "worker health check failed: worker_url=%s",
                base,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return {"ready": False, "loaded": False, "device": "unavailable"}

    def health(self) -> dict[str, dict[str, object]]:
        return {name: self._get_health(url) for name, url in self.urls.items()}

    def unload_all(self) -> dict[str, dict[str, object]]:
        results: dict[str, dict[str, object]] = {}
        for name, base in self.urls.items():
            try:
                response = self.client.post(f"{base}/internal/unload")
                response.raise_for_status()
                body = response.json()
                results[name] = {"unloaded": bool(body.get("unloaded", False))}
            except Exception as exc:
                logger.error(
                    "worker unload failed: worker=%s",
                    name,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                results[name] = {"unloaded": False, "error": "worker_unavailable"}
        return results

    def _post(self, url: str, data: bytes, parameters: dict[str, object]) -> bytes:
        try:
            with self.client.stream(
                "POST",
                url,
                params={
                    key: None if value is None else str(value) for key, value in parameters.items()
                },
                files={"file": ("input", data, "application/octet-stream")},
            ) as response:
                response.raise_for_status()
                declared = response.headers.get("content-length")
                if declared is not None and declared.isdigit():
                    if int(declared) > self.max_output_bytes:
                        raise WorkerUnavailable("worker output exceeds configured limit")
                output = bytearray()
                for chunk in response.iter_bytes():
                    if len(chunk) > self.max_output_bytes - len(output):
                        raise WorkerUnavailable("worker output exceeds configured limit")
                    output.extend(chunk)
                return bytes(output)
        except Exception as exc:
            raise WorkerUnavailable("worker request failed") from exc

    def upscale(self, data: bytes, **parameters: object) -> bytes:
        return self._post(f"{self.upscale_url}/internal/upscale", data, parameters)

    def background(self, data: bytes, **parameters: object) -> bytes:
        return self._post(f"{self.background_url}/internal/background-removal", data, parameters)


class PeerEvictor:
    """Call private peer unload controls while the caller already owns the global lane."""

    def __init__(self, peer_urls: tuple[str, ...], timeout_seconds: float = 30.0) -> None:
        self.peer_urls = tuple(url.rstrip("/") for url in peer_urls)
        self.timeout_seconds = timeout_seconds

    def __call__(self) -> None:
        for peer in self.peer_urls:
            try:
                response = httpx.post(
                    f"{peer}/internal/unload",
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                if response.json().get("unloaded") is not True:
                    raise ValueError("peer did not confirm unload")
            except Exception as exc:
                logger.error(
                    "peer model eviction failed: peer_url=%s",
                    peer,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                raise WorkerUnavailable("peer model eviction failed") from exc


class FakeWorkerClient:
    def __init__(self) -> None:
        self.model_invocations = 0
        self.model_loads = 0
        self.unload_calls = 0
        self.last_upscale: dict[str, object] = {}
        self.last_background: dict[str, object] = {}
        self._loaded: dict[str, str | None] = {
            "upscale": None,
            "background-removal": None,
            "generation": None,
        }

    def set_loaded(self, capability: str, model: str) -> None:
        self._loaded[capability] = model

    def health(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for capability, model in self._loaded.items():
            status: dict[str, object] = {
                "ready": True,
                "loaded": model is not None,
                "device": "cpu-test",
                "weightsAvailable": True,
            }
            if model is not None:
                status["loadedModel"] = model
            result[capability] = status
        return result

    def unload_all(self) -> dict[str, dict[str, object]]:
        self.unload_calls += 1
        for capability in self._loaded:
            self._loaded[capability] = None
        return {name: {"unloaded": True} for name in self._loaded}

    @staticmethod
    def _open(data: bytes) -> Image.Image:
        with Image.open(BytesIO(data)) as image:
            return image.copy()

    def upscale(self, data: bytes, **parameters: object) -> bytes:
        self.model_invocations += 1
        self.last_upscale = parameters
        opened = self._open(data)
        image = opened.convert("RGBA" if opened.mode == "RGBA" else "RGB")
        scale_value = parameters["outscale"]
        if not isinstance(scale_value, (int, float)):
            raise ValueError("fake worker outscale must be numeric")
        scale = float(scale_value)
        image = image.resize((round(image.width * scale), round(image.height * scale)))
        output = BytesIO()
        image.save(output, "PNG")
        return output.getvalue()

    def background(self, data: bytes, **parameters: object) -> bytes:
        self.model_invocations += 1
        self.last_background = parameters
        image = self._open(data).convert("RGBA")
        output = BytesIO()
        image.save(output, "PNG")
        return output.getvalue()
