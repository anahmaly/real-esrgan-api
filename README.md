# image-api

Private-LAN image gateway with process-isolated GPU workers for upscaling, background removal, Ideogram 4 text generation, and LongCat single-image editing.

## Architecture and GPU residency

Only the `image-api` gateway publishes port `8000`. Worker control routes are Compose-internal only. Every real operation owns the shared `/state/gpu-lane.lock` through peer eviction, model loading, inference, and post-processing. A gateway/client disconnect therefore cannot release the GPU lane while native work continues.

Before a selected worker loads or runs a model, it asks every peer worker to unload resident models while it still owns the global lane. Peer eviction fails closed: an unreachable peer prevents the new model from loading. The generation worker also unloads on switches among `ideogram-4-nf4`, `longcat-image-edit`, and `longcat-image-edit-turbo`. Same-worker/same-model requests may reuse the resident pipeline. Disposal removes Diffusers/Accelerate hooks, clears model references, runs garbage collection, and conditionally clears the CUDA allocator.

Generation and edit admissions use synchronous SQLite durability. Every shared-state writer runs as numeric UID/GID `10001:10001`. The root-only `state-init` one-shot performs a no-symlink, same-volume ownership migration capped by `IMAGE_API_STATE_INIT_MAX_ENTRIES`; all writers wait for it to finish. Gateway readiness then proves that the source lock/file path and SQLite are writable without loading or invoking a model. Image-edit source bytes are validated, normalized to a deterministic RGB PNG, fsynced, and atomically published under `/state/sources` before `202` is returned. The durable idempotency fingerprint includes the exact uploaded-byte SHA-256 and every semantic parameter. Results are validated RGB PNGs, fsynced, atomically renamed, and then marked successful. Restart reconciliation never re-invokes an interrupted task.

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

### Upscale and staged square clip-art processing

Model IDs remain `RealESRGAN_x4plus` and `RealESRGAN_x4plus_anime_6B`. `outscale` remains `1–4`; `tile` remains `0` or a multiple of 32 through 1024. Real-ESRGAN always receives RGB: alpha is discarded rather than upscaled. Inputs larger than 1024 on either edge are forced to the worker's 512-pixel tile when the compatibility value `tile=0` is supplied.

The supported 8K-square clip-art order is explicit and is not collapsed into one request:

1. Upload a `1024x1024` RGB image with `outscale=4`; require the response to be exactly `4096x4096` RGB.
2. Upload that `4096x4096` RGB response with `outscale=2`; require the response to be exactly `8192x8192` RGB (`67,108,864` pixels).
3. Upload the `8192x8192` RGB response once to BiRefNet background removal; require exactly `8192x8192` RGBA PNG.

```sh
curl -f -X POST \
  'http://HOST:8000/v1/upscale?model=RealESRGAN_x4plus&outscale=4&tile=512' \
  -F 'file=@clipart-1024-rgb.png' -o clipart-4096-rgb.png
curl -f -X POST \
  'http://HOST:8000/v1/upscale?model=RealESRGAN_x4plus&outscale=2&tile=512' \
  -F 'file=@clipart-4096-rgb.png' -o clipart-8192-rgb.png
```

The gateway and worker reject a dimension or mode mismatch; they never silently substitute a 4K result for a failed 8K request. "4K" here means square `4096x4096`, and "8K" means square `8192x8192`, not UHD landscape.

### Background removal

Model IDs remain `bria-rmbg-2.0` and `birefnet-hr-matting`. For the staged clip-art path, use BiRefNet once after both RGB upscales:

```sh
curl -f -X POST \
  'http://HOST:8000/v1/background-removal?model=birefnet-hr-matting&birefnet_inference_size=4096' \
  -F 'file=@clipart-8192-rgb.png' -o clipart-8192-rgba.png
```

The output canvas remains `8192x8192`, but `birefnet_inference_size` is independently bounded to `512–4096` and defaults to `2048`. BiRefNet therefore performs model inference at no more than its documented 4096 internal size and post-processes back to the original canvas. This is not native 8K matting.

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

The established generation/image-edit admission limits remain unchanged:

- `IMAGE_API_MAX_REQUEST_BYTES=21000000` caps non-processing request bodies, including `/v1/image-edits`.
- `IMAGE_API_MAX_UPLOAD_BYTES=20000000`, `IMAGE_API_MAX_INPUT_WIDTH=10000`, `IMAGE_API_MAX_INPUT_HEIGHT=10000`, and `IMAGE_API_MAX_INPUT_PIXELS=40000000` retain the existing edit upload and decoded-image contract.
- `IMAGE_API_MAX_OUTPUT_PIXELS=80000000`, `IMAGE_API_MAX_DECODED_INPUT_BYTES=160000000`, and `IMAGE_API_MAX_DECODED_OUTPUT_BYTES=320000000` keep non-processing decoded memory bounded.

The synchronous `/v1/upscale` and `/v1/background-removal` routes use separate 8K processing limits:

- `IMAGE_API_PROCESSING_MAX_REQUEST_BYTES=285000000` caps the complete processing multipart request.
- `IMAGE_API_PROCESSING_MAX_UPLOAD_BYTES=280000000` caps the encoded processing upload at both gateway and worker.
- `IMAGE_API_PROCESSING_MAX_ENCODED_OUTPUT_BYTES=300000000` independently caps processing-worker HTTP output and BiRefNet PNG post-processing.
- `IMAGE_API_PROCESSING_MAX_INPUT_WIDTH=8192`, `IMAGE_API_PROCESSING_MAX_INPUT_HEIGHT=8192`, `IMAGE_API_PROCESSING_MAX_INPUT_PIXELS=67108864`, and `IMAGE_API_PROCESSING_MAX_OUTPUT_PIXELS=67108864` define the square-8K canvas contract.
- `IMAGE_API_PROCESSING_MAX_DECODED_INPUT_BYTES=268435456` and `IMAGE_API_PROCESSING_MAX_DECODED_OUTPUT_BYTES=268435456` conservatively budget four decoded bytes per processing pixel.
- `IMAGE_API_PROCESSING_MAX_NATIVE_WIDTH=16384`, `IMAGE_API_PROCESSING_MAX_NATIVE_HEIGHT=16384`, and `IMAGE_API_PROCESSING_MAX_NATIVE_PIXELS=268435456` separately bound Real-ESRGAN's native intermediate canvas.
- `IMAGE_API_PROCESSING_MAX_NATIVE_BYTES=3221225472` bounds that native RGB intermediate at three float32 channels per pixel; it is an admission budget, not a promise about total allocator usage.
- `IMAGE_API_WORKER_TIMEOUT_SECONDS=900` is the gateway-to-worker processing timeout. Increase it explicitly if a measured 8K run needs longer; timeout never changes requested dimensions.
- `IMAGE_API_LANE_TIMEOUT_SECONDS` bounds waiting to enter the singleton GPU lane; it does not interrupt work that already owns the lane.
- `IMAGE_API_MAX_QUEUE_DEPTH=100` bounds durable generation/edit admission.
- `IMAGE_API_STATE_INIT_MAX_ENTRIES=100000` bounds the existing-volume ownership migration and fails startup rather than partially claiming success when exceeded.

An explicit slower-host example is:

```sh
IMAGE_API_WORKER_TIMEOUT_SECONDS=1800 \
IMAGE_API_LANE_TIMEOUT_SECONDS=300 \
docker compose up -d
```

These are independent controls rather than an encoded-size proxy for decoded memory. Gateway uploads remain in Starlette's spooled upload file, processing-worker uploads and gateway worker responses spool to disk after 8 MiB, and gateway responses stream from that bounded spool. The pinned Real-ESRGAN x4 helper applies `outscale=2` only after producing its native x4 result: the 4096-to-8192 stage therefore assembles a `16384x16384` intermediate before Lanczos downsampling. Its RGB float32 representation alone is 3 GiB, in addition to half/float tensors, the encoded input, the final 8192 image, model state, and allocator overhead. The pinned `rembg-api` adapter likewise requires in-process image/Pillow buffers and encoded `bytes`; an 8192 RGB canvas is about 192 MiB and an RGBA canvas is 256 MiB before overhead. Operators must budget host RAM, temporary-disk space, and VRAM for these unavoidable peaks.

Square 4K remains the established dependable path. The deterministic suite proves contracts, boundaries, wiring, and exact dimensions with lightweight fixtures, but it does not prove real square-8K quality, VRAM headroom, or latency. A live RTX 4090 acceptance run is still required before calling the 8K path operationally validated.

Accepted edit source formats remain PNG, JPEG, and WebP in common grayscale/palette/RGB/RGBA modes. They are normalized to RGB PNG only after exact uploaded-byte hashing.

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
