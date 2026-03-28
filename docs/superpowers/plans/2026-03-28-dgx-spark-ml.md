# DGX Spark ML Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a custom Docker image that runs Immich ML with GPU acceleration on DGX Spark (ARM64 Grace + NVIDIA Blackwell GB10, CUDA 13.0).

**Architecture:** Three-stage Docker multi-stage build: (1) build onnxruntime-gpu from source with SM_120 + CUDA 13, (2) install Immich ML Python dependencies + custom ORT wheel, (3) minimal production image with CUDA runtime + cuDNN + Python venv.

**Tech Stack:** Docker multi-stage build, nvidia/cuda 13.0.2 arm64 images, onnxruntime 1.24.4 (source build), Python 3.11, uv package manager, cuDNN 9.x

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `machine-learning/Dockerfile.dgx-spark` | Create | Three-stage Dockerfile for arm64 CUDA build |
| `docker/docker-compose.dgx-spark.yml` | Create | Compose override that swaps ML service to local image |

No changes to any existing Immich source files.

---

### Task 1: Create Dockerfile.dgx-spark

**Files:**
- Create: `machine-learning/Dockerfile.dgx-spark`

- [ ] **Step 1: Create Stage 1 – onnxruntime-gpu source build**

```dockerfile
# ==============================================================================
# Stage 1: Build onnxruntime-gpu from source for ARM64 + Blackwell (SM_120)
# ==============================================================================
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS builder-ort

# renovate: datasource=github-tags depName=microsoft/onnxruntime
ARG ORT_VERSION="v1.24.4"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    cmake git g++ && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN git clone --depth 1 --branch ${ORT_VERSION} --recurse-submodules --shallow-submodules \
    https://github.com/microsoft/onnxruntime.git /onnxruntime

WORKDIR /onnxruntime

# Patch: Remove 120 from ARCHITECTURES_WITH_ACCEL to avoid building sm_120a
# (accelerated variant that doesn't work on Blackwell consumer/DGX GPUs).
# The build will still target sm_120 via CMAKE_CUDA_ARCHITECTURES.
RUN sed -i 's/set(ARCHITECTURES_WITH_ACCEL "90" "100" "101" "120")/set(ARCHITECTURES_WITH_ACCEL "100" "101")/' \
    cmake/CMakeLists.txt || true && \
    grep -r 'ARCHITECTURES_WITH_ACCEL' cmake/ | head -5

RUN ./build.sh \
    --config Release \
    --build_wheel \
    --build_shared_lib \
    --use_cuda \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr \
    --cuda_version 13.0 \
    --parallel \
    --cmake_extra_defines \
      CMAKE_CUDA_ARCHITECTURES=120 \
      onnxruntime_USE_FLASH_ATTENTION=OFF \
    --skip_tests

# Collect the built wheel
RUN mkdir /ort-wheel && cp build/Linux/Release/dist/onnxruntime_gpu-*.whl /ort-wheel/
```

- [ ] **Step 2: Create Stage 2 – Python venv with Immich ML dependencies**

```dockerfile
# ==============================================================================
# Stage 2: Install Immich ML Python dependencies + custom ORT wheel
# ==============================================================================
FROM python:3.11-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends g++ && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

# Install all deps with the cpu extra first (onnxruntime from PyPI, arm64-compatible)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --extra cpu --no-dev --no-editable --no-install-project --compile-bytecode --no-progress --active --link-mode copy

# Replace onnxruntime (cpu) with the custom-built onnxruntime-gpu wheel
COPY --from=builder-ort /ort-wheel /ort-wheel
RUN /opt/venv/bin/pip install --no-deps --force-reinstall /ort-wheel/onnxruntime_gpu-*.whl && \
    rm -rf /ort-wheel
```

- [ ] **Step 3: Create Stage 3 – Production image**

```dockerfile
# ==============================================================================
# Stage 3: Production image
# ==============================================================================
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04 AS prod

# Install Python 3.11 from builder
COPY --from=builder /usr/local/bin/python3 /usr/local/bin/python3
COPY --from=builder /usr/local/bin/python3.11 /usr/local/bin/python3.11
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/lib/libpython3.11.so* /usr/local/lib/
RUN ldconfig

ENV LD_PRELOAD=/usr/lib/libmimalloc.so.2 \
    MACHINE_LEARNING_MODEL_ARENA=false

RUN apt-get update && \
    apt-get install -y --no-install-recommends tini ccache libgl1 libglib2.0-0 libgomp1 libmimalloc2.0 && \
    apt-get autoremove -yqq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s "/usr/lib/$(arch)-linux-gnu/libmimalloc.so.2" /usr/lib/libmimalloc.so.2

WORKDIR /usr/src
ENV TRANSFORMERS_CACHE=/cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/usr/src \
    DEVICE=cuda \
    VIRTUAL_ENV=/opt/venv \
    MACHINE_LEARNING_CACHE_FOLDER=/cache

# Prevent core dumps
RUN echo "hard core 0" >> /etc/security/limits.conf && \
    echo "fs.suid_dumpable 0" >> /etc/sysctl.conf && \
    echo 'ulimit -S -c 0 > /dev/null 2>&1' >> /etc/profile

COPY --from=builder /opt/venv /opt/venv
COPY scripts/healthcheck.py .
COPY immich_ml immich_ml

ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "immich_ml"]

HEALTHCHECK CMD python3 healthcheck.py
```

- [ ] **Step 4: Assemble the full Dockerfile**

Write the three stages into a single file `machine-learning/Dockerfile.dgx-spark` in the order above (Stage 1, Stage 2, Stage 3).

- [ ] **Step 5: Commit**

```bash
git add machine-learning/Dockerfile.dgx-spark
git commit -m "feat(ml): add Dockerfile for DGX Spark (ARM64 + Blackwell GB10)"
```

---

### Task 2: Create docker-compose override

**Files:**
- Create: `docker/docker-compose.dgx-spark.yml`

- [ ] **Step 1: Create the compose override file**

```yaml
# Override for DGX Spark: builds ML image locally with ARM64 CUDA support
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

- [ ] **Step 2: Commit**

```bash
git add docker/docker-compose.dgx-spark.yml
git commit -m "feat(ml): add docker-compose override for DGX Spark"
```

---

### Task 3: Build and verify the image on DGX Spark

- [ ] **Step 1: Build the image**

Run on the DGX Spark:

```bash
cd machine-learning
docker build -f Dockerfile.dgx-spark -t immich-ml-dgx-spark:latest .
```

This will take 30-60 minutes (onnxruntime source build). Monitor for:
- CUDA toolkit detection in ORT build
- SM_120 appearing in cmake output
- Wheel creation in `build/Linux/Release/dist/`
- Successful Python/venv assembly in Stage 2

- [ ] **Step 2: Quick smoke test – check ORT providers**

```bash
docker run --rm --gpus all immich-ml-dgx-spark:latest \
  python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected output should include `CUDAExecutionProvider`:
```
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

- [ ] **Step 3: Start the full stack**

```bash
cd docker
docker compose -f docker-compose.yml -f docker-compose.dgx-spark.yml up -d
```

- [ ] **Step 4: Verify ML service health**

```bash
docker logs immich_machine_learning 2>&1 | head -30
```

Look for:
- `Setting execution providers to ['CUDAExecutionProvider', 'CPUExecutionProvider']`
- No CUDA errors or fallback-to-CPU warnings
- Healthcheck passing: `docker inspect --format='{{.State.Health.Status}}' immich_machine_learning`

- [ ] **Step 5: End-to-end test**

Upload a photo via the Immich web UI. Check that:
- Smart search / CLIP embedding is generated
- Face detection runs
- No errors in ML container logs

---

## Troubleshooting Reference

| Problem | Fix |
|---------|-----|
| `builder-ort` fails at cmake with "unsupported CUDA architecture 120" | ORT version too old. Try building from `main` branch instead of tag. |
| `ARCHITECTURES_WITH_ACCEL` sed patch doesn't match | The variable may have moved. `grep -r ARCHITECTURES_WITH_ACCEL cmake/` to find current location and patch manually. |
| ORT build fails with OOM | Reduce parallelism: add `--parallel 4` (or lower) to build.sh |
| `CUDAExecutionProvider` not in available providers | cuDNN missing or wrong version. Verify with `python3 -c "import ctypes; ctypes.cdll.LoadLibrary('libcudnn.so')"` inside container. |
| Python 3.11 not found in Ubuntu 24.04 | Ubuntu 24.04 ships Python 3.12. Install 3.11 via `deadsnakes` PPA: `add-apt-repository ppa:deadsnakes/ppa && apt-get install python3.11 python3.11-dev` |
| `libmimalloc.so.2` not found | Package may be named `libmimalloc2.0` or `libmimalloc2` depending on Ubuntu version. Check with `apt-cache search mimalloc`. |
