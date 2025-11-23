import io
import cv2
import numpy as np
import torch

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

app = FastAPI()

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
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=(device.type == "cuda"),  # use half precision on GPU
    device=device,
)

# ---------------- API endpoint ----------------

@app.post("/upscale/")
async def upscale_image(
    file: UploadFile = File(...),
    outscale: float = 2.0,  # default 2x; you can send 4.0, etc.
):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")

        # Decode image from bytes
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Run Real-ESRGAN
        output, _ = upsampler.enhance(img, outscale=outscale)

        # Encode result to PNG bytes
        ok, buf = cv2.imencode(".png", output)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode output image")

        return StreamingResponse(
            io.BytesIO(buf.tobytes()),
            media_type="image/png",
        )

    except HTTPException:
        raise
    except Exception as e:
        # Shows up in container logs
        print("Upscale error:", repr(e))
        raise HTTPException(status_code=500, detail="Upscale failed internally")
