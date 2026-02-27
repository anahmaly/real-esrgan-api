# Real-ESRGAN API (GPU-Enabled)

A FastAPI-based API wrapper around **Real-ESRGAN**, packaged in Docker with full **NVIDIA GPU support** and optimized for in-memory image processing.

This fork includes several reliability improvements, including:
- proper CUDA-enabled PyTorch installation
- GPU device detection at runtime
- in-memory processing (no temporary files)
- simplified `/upscale/` endpoint
- correct multipart form field usage (`file=`)

Upstream Real-ESRGAN project:
https://github.com/xinntao/Real-ESRGAN

---

## 🚀 Getting Started

### 🛠 Build the Docker Image

```sh
docker build -t real-esrgan-api .
```

### ▶️ Run the API with NVIDIA GPU Support

Make sure you have installed:

- NVIDIA drivers
- `nvidia-container-toolkit`
- Docker configured for GPU passthrough (`--gpus all`)

Run the container:

```sh
docker run -d --gpus all -p 8000:8000 --name realesrgan real-esrgan-api
```

### ▶️ Or Use Docker Compose

This repo ships with `compose.yml` configured for GPU (`gpus: all`).

```sh
docker compose up -d --build
```

This will start the API and expose it on **http://localhost:8000**.

### ✅ Verify GPU is visible inside the container

```sh
docker exec -it realesrgan python -c "import torch; print('cuda:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count())"
```

Expected output includes `cuda: True` and a device count `>= 1`.

---

## 🔗 API Endpoint

The upscaling endpoint is:

```
POST http://localhost:8000/upscale/
```

### 📤 Example Request

Note: the form field must be named **`file`** (not `image`).

```sh
curl -X POST \
     -F "file=@path/to/image.jpg" \
     -o output.png \
     http://localhost:8000/upscale/
```

The API returns a PNG-formatted upscaled image.

### Optional Parameters

You can adjust upscale factor:

```sh
curl -X POST \
     -F "file=@input.jpg" \
     -F "outscale=4" \
     -o output.png \
     http://localhost:8000/upscale/
```

Default is **2×**.

---

## 🧠 How It Works

- The Real-ESRGAN model is loaded once on startup.
- If a GPU is available, the model runs on CUDA with half-precision.
- Uploads are decoded in memory (no temp files).
- Results are returned as a streaming PNG response.

This avoids issues with deleted temp files and is much faster.

---

## 🛠 Requirements

- Docker (with Compose plugin)
- NVIDIA GPU
- NVIDIA container toolkit (`nvidia-ctk`)
- CUDA-supported GPU drivers
- Internet access for initial model download (or place your own `.pth` model in `/Real-ESRGAN/weights/`)

### Troubleshooting (Docker GPU)

If `torch.cuda.is_available()` is `False` in-container:

1. Verify host GPU runtime works:

```sh
docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi
```

2. Ensure NVIDIA container toolkit is installed/configured:

```sh
nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

3. Rebuild and restart this service:

```sh
docker compose down
docker compose up -d --build
```

---

## 📸 Preview

![Real-ESRGAN API Preview](https://github.com/natnael9402/real-esrgan-api/blob/main/1.png)

---

## 📝 License

This project is licensed under the MIT License.

---

Made with ❤️

Forked & enhanced for GPU support and in-memory processing
