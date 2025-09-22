# ===== 基础镜像：CUDA 12.1 + 构建工具 =====
#################################################################
# Multi-stage Dockerfile optimized for minimal runtime image size
# Builder stage: use devel image to install build deps and create venv
# Runtime stage: use cuda runtime image and copy only the venv + minimal OS libs
#################################################################

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Singapore \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-distutils python3-pip python3-dev \
    build-essential ca-certificates cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip wheel setuptools

# Create a venv and install Python packages there to allow copying into a smaller runtime image
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only the Python packages actually used by this repo
# - Drop torchvision/torchaudio and nerfstudio to reduce size/time
# - Add open3d (required by manual_splat GUI)
RUN pip install --no-cache-dir \
    torch==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
    numpy==1.26.4 \
    imageio==2.34.0 \
    transforms3d \
    tqdm \
    pyyaml \
    scipy \
    plyfile \
    open3d==0.18.0 \
    gsplat

#################################################################
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Singapore \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install only minimal OS packages required at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg libglib2.0-0 libgl1 libegl1 libxext6 libxrender1 \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project source into the image
WORKDIR /workspace
COPY . /workspace

WORKDIR /workspace/src

# Optional useful runtime envs
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

CMD ["/bin/bash"]
