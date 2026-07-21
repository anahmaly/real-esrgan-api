# image-api

Private-LAN image gateway with process-isolated GPU workers for upscaling, background removal, Ideogram 4 text generation, and LongCat single-image editing.

## Architecture and GPU residency

Only the `image-api` gateway publishes port `8000`. Worker control routes are Compose-internal only. Every real operation owns the shared `/state/gpu-lane.lock` through peer eviction, model loading, inference, and post-processing. A gateway/client disconnect therefore cannot release the GPU lane while native work continues.

Before a selected worker loads or runs a model, it asks every peer worker to unload resident models while it still owns the global lane. Peer eviction fails closed: an unreachable peer prevents the new model from loading. The generation worker also unloads on switches among `ideogram-4-nf4`, `longcat-image-edit`, and `longcat-image-edit-turbo`. Same-worker/same-model requests may reuse the resident pipeline. Disposal removes Diffusers/Accelerate hooks, clears model references, runs garbage collection, and conditionally clears the CUDA allocator.

Generation and edit admissions use synchronous SQLite durability. Image-edit source bytes are validated, normalized to a deterministic RGB PNG, fsynced, and atomically published under `/state/sources` before `202` is returned. The durable idempotency fingerprint includes the exact uploaded-byte SHA-256 and every semantic parameter. Results are validated RGB PNGs, fsynced, atomically renamed, and then marked successful. Restart reconciliation never re-invokes an interrupted task.

## Public API

- `GET /health` — capability readiness, worker/weight state, and the loaded model; does not load a model.
- `GET /v1/models` — supported models and honest generation/editing contracts.
- `POST /v1/upscale` — synchronous multipart upscale.
- `POST /v1/background-removal` — synchronous multipart background removal.
- `POST /v1/generations` — durable asynchronous Ideogram generation.
- `GET /v1/generations/{taskId}` and `/image` — generation status/result.
- `POST /v1/image-edits` — durable asynchronous LongCat single-image edit.
- `GET /v1/image-edits/{taskId}` and `/image` — edit status/result.
- `POST /v1/models/unload` — lane-serialized unload of all resident models with bounded per-worker results.

OpenAPI is available at `/openapi.json` and `/docs`.

### Upscale

Model IDs are `RealESRGAN_x4plus` and `RealESRGAN_x4plus_anime_6B`. `outscale` is `1–4`; `tile` is `0` or a multiple of 32 through 1024.

```sh
curl -f -X POST \
  'http://HOST:8000/v1/upscale?model=RealESRGAN_x4plus&outscale=2&tile=512' \
  -F 'file=@input.png' -o output.png
```

### Background removal

Model IDs are `bria-rmbg-2.0` and `birefnet-hr-matting`.

```sh
curl -f -X POST \
  'http://HOST:8000/v1/background-removal?model=birefnet-hr-matting&birefnet_inference_size=2048' \
  -F 'file=@input.png' -o foreground.png
```

### Ideogram 4 generation (compatibility retained)

`ideogram-4-nf4` remains available with the existing `V4_QUALITY_48`, `V4_DEFAULT_20`, and `V4_TURBO_12` presets. Dimensions are multiples of 16 from 256 through 2048.

```sh
curl -f -X POST http://HOST:8000/v1/generations \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: product-123-revision-4' \
  -d '{
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "sampler_preset": "V4_DEFAULT_20",
    "structured_caption": {"description": "A blue ceramic bee on a clean white surface"}
  }'
```

Plain prompts retain the existing requirement for `magic_prompt=true`, `IMAGE_API_MAGIC_PROMPT_BACKEND`, and `IMAGE_API_MAGIC_PROMPT_API_KEY`.

### LongCat image editing

Supported IDs:

- `longcat-image-edit`: BF16, guidance `4.5`, `50` steps.
- `longcat-image-edit-turbo`: BF16, guidance `1.0`, `8` steps.

Both use `negative_prompt=''`, one source image, one output, and a CPU-seeded generator. These official defaults are fixed by the API rather than exposed as arbitrary overrides. The released pipeline accepts exactly one source image, preserves its aspect ratio, and targets approximately one megapixel. It exposes no denoising/edit-strength parameter.

```sh
curl -f -X POST http://HOST:8000/v1/image-edits \
  -H 'Idempotency-Key: listing-123-edit-1' \
  -F 'model=longcat-image-edit-turbo' \
  -F 'prompt=Replace the background with a clean pale blue studio backdrop' \
  -F 'negative_prompt=' \
  -F 'seed=43' \
  -F 'file=@source.png'
```

The `202` response contains `taskId`. Poll and retrieve the result:

```sh
curl -f http://HOST:8000/v1/image-edits/TASK_ID
curl -f http://HOST:8000/v1/image-edits/TASK_ID/image -o edited.png
```

Replaying the same idempotency key with the same upload and fields returns the original task. Changing the source bytes, model, prompt, negative prompt, or seed returns `409`.

### Explicit model unload

```sh
curl -f -X POST http://HOST:8000/v1/models/unload
```

The gateway owns the singleton GPU lane before contacting all three workers. It busy-fails with `503` if the configured lane timeout expires and returns `503` with bounded per-worker statuses if any worker does not confirm unload. Internal unload handlers never reacquire the lane.

## Local model acquisition and mounts

Production is offline: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, local-only Diffusers loading, and no request-time downloads. Model weights remain host data outside Git.

LongCat provenance:

- Source: `meituan-longcat/LongCat-Image@f0e4c43c5ef74b011ff71570fbfc2bdffbc9ab06`, Apache-2.0.
- Standard: `meituan-longcat/LongCat-Image-Edit@7b54ef423aa7854be7861600024be5c56ab7875a`.
- Turbo: `meituan-longcat/LongCat-Image-Edit-Turbo@6a7262de5549f0bf0ec54c08ef7d283ef41f3214`.

Acquire the ungated snapshots explicitly on the host (not during container startup):

```sh
mkdir -p models/longcat-image-edit models/longcat-image-edit-turbo
hf download meituan-longcat/LongCat-Image-Edit \
  --revision 7b54ef423aa7854be7861600024be5c56ab7875a \
  --local-dir models/longcat-image-edit
printf '%s\n' 7b54ef423aa7854be7861600024be5c56ab7875a \
  > models/longcat-image-edit/.image-api-revision
hf download meituan-longcat/LongCat-Image-Edit-Turbo \
  --revision 6a7262de5549f0bf0ec54c08ef7d283ef41f3214 \
  --local-dir models/longcat-image-edit-turbo
printf '%s\n' 6a7262de5549f0bf0ec54c08ef7d283ef41f3214 \
  > models/longcat-image-edit-turbo/.image-api-revision
```

Each checkpoint is approximately 29.3 GB on disk. Official CPU offload uses approximately 18–19 GB VRAM and explicitly supports a desktop RTX 4090 with 24 GB VRAM. Host system RAM must also accommodate the offloaded portions; upstream does not publish a fixed minimum, so leave substantial headroom. No LongCat quantization support is claimed or configured.

Host-path overrides:

- `IMAGE_API_LONGCAT_EDIT_WEIGHTS_HOST_PATH` (default `./models/longcat-image-edit`).
- `IMAGE_API_LONGCAT_EDIT_TURBO_WEIGHTS_HOST_PATH` (default `./models/longcat-image-edit-turbo`).
- `IMAGE_API_IDEOGRAM_WEIGHTS_HOST_PATH`, `IMAGE_API_UPSCALE_WEIGHTS_HOST_PATH`, `IMAGE_API_BRIA_WEIGHTS_HOST_PATH`, and `IMAGE_API_BIREFNET_WEIGHTS_HOST_PATH` retain their existing meanings.

The generation image pins and import-checks `torch==2.11.0`, `torchvision==0.26.0`, `diffusers==0.37.0`, `transformers==4.57.1`, `accelerate==1.11.0`, and `safetensors==0.6.2`, while retaining the official Ideogram adapter at `990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2`.

See `NOTICE.md` and `licenses/`. Model weights are not included in this repository.

## Build and start

```sh
# Validate first.
docker compose config

# Rebuild production images without starting them.
./scripts/rebuild-images.sh

# Start already-built GPU services and verify readiness/CUDA.
./scripts/run-gpu.sh
```

Equivalent direct Compose startup:

```sh
docker compose up -d --build
```

Only the gateway is host-published. Restrict `HOST:8000` with the host firewall to trusted LAN devices.

CPU-only deterministic wiring uses fake workers and no model/network inference:

```sh
docker compose -f compose.yml -f compose.test.yml up --build
```

## Configuration bounds

- `IMAGE_API_MAX_REQUEST_BYTES` default `21000000` for the entire request body.
- `IMAGE_API_MAX_UPLOAD_BYTES` default `20000000` with chunked upload reads.
- `IMAGE_API_MAX_INPUT_PIXELS` default `40000000`.
- `IMAGE_API_MAX_OUTPUT_PIXELS` default `80000000`.
- `IMAGE_API_MAX_QUEUE_DEPTH` default `100`.
- `IMAGE_API_WORKER_TIMEOUT_SECONDS` and `IMAGE_API_LANE_TIMEOUT_SECONDS` are positive bounded waits.

Accepted edit source formats are PNG, JPEG, and WebP in common grayscale/palette/RGB/RGBA modes. They are normalized to RGB PNG only after exact uploaded-byte hashing.

## Development

```sh
uv sync --extra test --locked
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy src/image_api src/image_api_workers
.venv/bin/pytest -q
docker compose config --quiet
docker compose -f compose.yml -f compose.test.yml config --quiet
```

Tests use fake pipelines/workers and deterministic local images. They do not download weights, run real inference, call model providers, or require a GPU.
