FROM python:3.8-slim-bullseye

# 1. System dependencies for OpenCV, git, wget, etc.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. Workdir for API
WORKDIR /app

# 3. Copy API requirements and install them (NO torch here)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Install CUDA-enabled PyTorch (adjust versions if needed)
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 5. Clone Real-ESRGAN and download weights
WORKDIR /
RUN git clone --depth 1 https://github.com/xinntao/Real-ESRGAN.git && \
    mkdir -p /Real-ESRGAN/weights && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
         -O /Real-ESRGAN/weights/RealESRGAN_x4plus.pth

# 6. Install Real-ESRGAN deps, but prevent it from overwriting torch / opencv
WORKDIR /Real-ESRGAN

#   - remove any torch + opencv-python lines from its requirements
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/opencv-python/d' requirements.txt && \
    pip install --no-cache-dir basicsr==1.4.2 facexlib==0.2.5 gfpgan==1.3.8 && \
    pip install --no-cache-dir -r requirements.txt && \
    # install headless OpenCV explicitly
    pip install --no-cache-dir opencv-python-headless && \
    python setup.py develop

# 7. Copy API code into /app and set workdir back
WORKDIR /app
COPY app/ .

EXPOSE 8000

# 8. Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
