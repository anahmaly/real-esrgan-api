import asyncio
import gc
import io

import cv2
import numpy as np
import torch

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

app = FastAPI()

DEFAULT_TILE = 512

# ---------------- Real-ESRGAN setup ----------------

# Pick device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4,
)

model_path = "/Real-ESRGAN/weights/RealESRGAN_x4plus.pth"

upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=DEFAULT_TILE,
    tile_pad=10,
    pre_pad=0,
    half=(device.type == "cuda"),  # use half precision on GPU
    device=device,
)

upsampler_lock = asyncio.Lock()

# ---------------- API endpoint ----------------

@app.post("/upscale/")
async def upscale_image(
    file: UploadFile = File(...),
    outscale: float = 2.0,  # default 2x; you can send 4.0, etc.
    tile: int = Query(
        DEFAULT_TILE,
        ge=0,
        description="Real-ESRGAN tile size. Use 0 to disable tiling.",
    ),
):
    data = None
    nparr = None
    img = None
    output = None
    buf = None

    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")

        # Decode image from bytes
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Run Real-ESRGAN without autograd graph allocation.
        # RealESRGANer stores tile size on the instance, so protect per-request
        # tile changes from concurrent requests sharing the global upsampler.
        async with upsampler_lock:
            previous_tile_size = upsampler.tile_size
            upsampler.tile_size = tile
            try:
                with torch.inference_mode():
                    output, _ = upsampler.enhance(img, outscale=outscale)
            finally:
                upsampler.tile_size = previous_tile_size

        # Encode result to PNG bytes
        ok, buf = cv2.imencode(".png", output)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode output image")

        result_bytes = buf.tobytes()
        return StreamingResponse(
            io.BytesIO(result_bytes),
            media_type="image/png",
        )

    except HTTPException:
        raise
    except Exception as e:
        # Shows up in container logs
        print("Upscale error:", repr(e))
        raise HTTPException(status_code=500, detail="Upscale failed internally")
    finally:
        # Release large arrays/tensors quickly between requests.
        del data, nparr, img, output, buf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
