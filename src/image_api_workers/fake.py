from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from PIL import Image

app = FastAPI(title="image-api-test-worker", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict[str, object]:
    return {"ready": True, "loaded": False, "device": "cpu-test"}


@app.post("/internal/unload")
def unload() -> dict[str, object]:
    return {"unloaded": True}


@app.post("/internal/upscale")
async def upscale(file: UploadFile, outscale: float, model: str, tile: int) -> Response:
    data = await file.read()
    with Image.open(BytesIO(data)) as image:
        output_image = image.resize((round(image.width * outscale), round(image.height * outscale)))
        output = BytesIO()
        output_image.save(output, "PNG")
    return Response(output.getvalue(), media_type="image/png")


@app.post("/internal/background-removal")
async def background(file: UploadFile) -> Response:
    data = await file.read()
    with Image.open(BytesIO(data)) as image:
        output = BytesIO()
        image.convert("RGBA").save(output, "PNG")
    return Response(output.getvalue(), media_type="image/png")
