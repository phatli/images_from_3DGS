# ===== 基础镜像：CUDA 12.1 + 构建工具 =====
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Singapore \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    FORCE_CUDA=1

# ---- 系统依赖 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates wget \
    build-essential cmake ninja-build \
    ffmpeg \
    libglib2.0-0 libgl1 libegl1 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Python & PyTorch（CUDA 12.1 对应预编译轮子）----
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip wheel setuptools

# PyTorch + CUDA 12.1（官方索引）
RUN pip install --no-cache-dir \
    torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ---- 基础科学栈 & 常用依赖 ----
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    imageio==2.34.0 \
    transforms3d \
    matplotlib \
    tqdm pyyaml ninja \
    scipy \
    plyfile

# ---- 安装 nerfstudio（含 3DGS 渲染功能、viewer、ns-render）----
# 可固定版本号，或安装主分支稳定版本。为简洁起见用 PyPI。
RUN pip install --no-cache-dir "nerfstudio[all]==1.1.3"


# gsplat
RUN pip install --no-cache-dir \
    torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 安装稳定版 gsplat
RUN pip install --no-cache-dir gsplat

# ---- 工作目录 ----
WORKDIR /workspace

# ---- 环境变量（可选优化）----
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# ---- 默认命令 ----
CMD ["/bin/bash"]
