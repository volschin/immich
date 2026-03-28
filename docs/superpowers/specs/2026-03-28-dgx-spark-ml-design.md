# Immich ML on DGX Spark (ARM64 + NVIDIA GB10/Blackwell)

## Problem

Immich's machine learning Docker images don't support ARM64 + NVIDIA GPU. The DGX Spark combines a Grace ARM64 CPU with a Blackwell GB10 GPU (CUDA 13.0, Driver 580.142, SM_120). The existing `cuda` variant is x86_64-only (`nvidia/cuda:12.2.2-runtime-ubuntu22.04`) and the `onnxruntime-gpu` PyPI wheel only targets x86_64.

## Goal

Build a custom Docker image that runs Immich ML on DGX Spark with GPU acceleration via CUDA. Private solution first, upstream-ready design.

## Constraints

- CUDA 13.0 / Driver 580.142 on host
- Blackwell GPU architecture: SM_120
- `onnxruntime-gpu` must be built from source (no arm64 PyPI wheel)
- Must be compatible with existing Immich docker-compose setup (only ML service replaced)
- Python 3.11 (Immich requirement)
- onnxruntime >= 1.23.2 (Immich requirement)

## Architecture

Three-stage Docker multi-stage build in `Dockerfile.dgx-spark`:

```
Stage 1: builder-ort
  Base: nvidia/cuda arm64 devel image (13.0 from Docker Hub or nvcr.io)
  Purpose: Build onnxruntime-gpu from source with CUDA + SM_120
  Output: Python wheel (.whl)

Stage 2: builder
  Base: python:3.11-bookworm (arm64)
  Purpose: Install Immich ML Python dependencies via uv, then install ort wheel from Stage 1
  Output: /opt/venv with all dependencies

Stage 3: prod
  Base: nvidia/cuda arm64 runtime image (13.0 from Docker Hub or nvcr.io)
  Purpose: Production image with Python, cuDNN, venv, and Immich ML code
  Entrypoint: tini + python -m immich_ml
```

## Stage Details

### Stage 1: builder-ort

```dockerfile
FROM nvidia/cuda:13.0.x-devel-ubuntu24.04 AS builder-ort
# Fallback: nvcr.io/nvidia/cuda:13.0.x-devel-ubuntu24.04
```

Build steps:
1. Install build dependencies: `python3.11-dev`, `cmake`, `git`, `g++`
2. Clone `microsoft/onnxruntime` at a tag >= 1.23.2 that includes Blackwell support (may need `main` or a nightly if 1.23.x doesn't support SM_120)
3. Build with:
   ```
   ./build.sh --config Release \
     --build_wheel \
     --use_cuda \
     --cuda_home /usr/local/cuda \
     --cudnn_home /usr \
     --parallel \
     --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=120 \
     --skip_tests
   ```
4. Output: `/build/dist/onnxruntime_gpu-*.whl`

Build time estimate: 30-60 minutes on DGX Spark.

### Stage 2: builder

```dockerfile
FROM python:3.11-bookworm AS builder
```

Steps:
1. Install `g++` (needed for some Python packages with native extensions)
2. Use `uv sync --frozen --extra cuda` to install all Immich ML dependencies except onnxruntime-gpu
3. Install the ort wheel from Stage 1: `pip install /ort/*.whl`

Note: `uv sync` will attempt to install `onnxruntime-gpu` from PyPI and fail on arm64. Solution: run `uv sync` with the `cpu` extra first (installs `onnxruntime`), then replace with the custom GPU wheel. Alternatively, override the dependency at install time.

### Stage 3: prod

```dockerfile
FROM nvidia/cuda:13.0.x-runtime-ubuntu24.04 AS prod
# Fallback: nvcr.io/nvidia/cuda:13.0.x-runtime-ubuntu24.04
```

Steps:
1. Install runtime dependencies: `tini`, `libgl1`, `libglib2.0-0`, `libgomp1`, `libmimalloc2.0`, `ccache`
2. Install `cuDNN 9.x` for arm64 (from NVIDIA repos)
3. Copy Python 3.11 from builder stage
4. Copy `/opt/venv` from builder stage
5. Copy Immich ML source code
6. Set environment variables matching upstream (DEVICE=cuda, paths, etc.)

## Files to Create

| File | Purpose |
|------|---------|
| `machine-learning/Dockerfile.dgx-spark` | Multi-stage Dockerfile |
| `docker/docker-compose.dgx-spark.yml` | Compose override, replaces ML service image |

### docker-compose.dgx-spark.yml

```yaml
services:
  immich-machine-learning:
    build:
      context: ../machine-learning
      dockerfile: Dockerfile.dgx-spark
    image: immich-ml-dgx-spark:latest
    extends:
      file: hwaccel.ml.yml
      service: cuda
```

Usage: `docker compose -f docker-compose.yml -f docker-compose.dgx-spark.yml up -d`

## Build & Deploy

```bash
# Build on the DGX Spark itself (simplest, native arm64)
cd machine-learning
docker build -f Dockerfile.dgx-spark -t immich-ml-dgx-spark:latest .

# Or use compose
cd docker
docker compose -f docker-compose.yml -f docker-compose.dgx-spark.yml build immich-machine-learning
docker compose -f docker-compose.yml -f docker-compose.dgx-spark.yml up -d
```

## Integration

No changes to Immich ML application code. The `OrtSession` class in `immich_ml/sessions/ort.py` already:
- Auto-detects `CUDAExecutionProvider` from available ORT providers
- Configures `arena_extend_strategy` and `device_id` for CUDA
- Falls back to CPU if CUDA unavailable

The `hwaccel.ml.yml` CUDA config (nvidia driver reservation) works unchanged for the DGX Spark.

## Risks & Open Points

1. **CUDA 13.0 base image availability**: Check Docker Hub first, fallback to `nvcr.io/nvidia/cuda` registry
2. **onnxruntime Blackwell support**: SM_120 support may require onnxruntime >= 1.24 or building from `main`. Must verify which version/commit includes Blackwell arch support.
3. **cuDNN version**: Must match CUDA 13.0 arm64. NVIDIA's `libcudnn9` packages should be available via apt from their repos.
4. **Build cache**: The onnxruntime build is expensive. Consider caching the wheel or the builder-ort stage via `docker buildx --cache-to`.
5. **onnxruntime dependency conflict**: `uv sync --extra cuda` pulls `onnxruntime-gpu` from PyPI (x86 only). Must handle this gracefully in the builder stage (install cpu extra, then override with custom wheel).

## Future Upstream Path

The design maps cleanly to a new device variant in Immich's build matrix:
- New `DEVICE=cuda-arm64` option
- `builder-cuda-arm64` stage with onnxruntime source build
- `prod-cuda-arm64` stage with arm64 CUDA runtime base
- CI matrix entry: `{device: cuda-arm64, platforms: linux/arm64, runs-on: arm64-runner}`
- New optional dependency in `pyproject.toml`: `cuda-arm64 = ["onnxruntime-gpu>=1.23.2,<2"]` (installed from custom-built wheel)
