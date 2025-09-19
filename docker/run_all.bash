#!/usr/bin/env bash
set -euo pipefail

# ========= 默认参数（可通过命令行覆盖） =========
SRC_HOST_DIR="${SRC_HOST_DIR:-$(pwd)/../src}"         # Host 上源码目录，默认 "../src"
CONTAINER_NAME="${CONTAINER_NAME:-gsr}"
IMAGE="${IMAGE:-stephenmao0927/gsr:v2}"

GAUSSIANS="${GAUSSIANS:-/workspace/src/data/point_cloud_gs.ply}"
IMG_PATHS="${IMG_PATHS:-/workspace/src/manual_splat/manual_out_img}"
OUTDIR="${OUTDIR:-/workspace/src/manual_splat/manual_out_poses}"

FOV="${FOV:-70}"
IMGW="${IMGW:-1920}"
IMGH="${IMGH:-1080}"
NEAR="${NEAR:-0.1}"
FAR="${FAR:-20.0}"

MIN_VISIBLE="${MIN_VISIBLE:-800}"
OVERLAP_RATIO="${OVERLAP_RATIO:-0.4}"
DS="${DS:-0.3}"

# ========= 命令行参数解析 =========
print_help() {
  cat <<EOF
Usage: $(basename "$0") [options]

# 路径（容器内路径）
  --gaussians PATH           默认: $GAUSSIANS
  --img-paths PATH           默认: $IMG_PATHS
  --outdir PATH              默认: $OUTDIR

# 相机参数
  --fov N                    默认: $FOV
  --imgw N                   默认: $IMGW
  --imgh N                   默认: $IMGH
  --near F                   默认: $NEAR
  --far F                    默认: $FAR

# 采样与筛选
  --min-visible N            默认: $MIN_VISIBLE
  --overlap-ratio F          默认: $OVERLAP_RATIO
  --ds F                     默认: $DS

# Docker 相关
  --src HOST_DIR             Host 上源码目录（映射到 /workspace/src），默认: $SRC_HOST_DIR
  --name NAME                容器名，默认: $CONTAINER_NAME
  --image IMAGE              镜像名:tag，默认: $IMAGE

示例：
  $(basename "$0") --gaussians /workspace/src/data/xyz.ply --fov 75 --imgw 2560 --imgh 1440
EOF
  exit 0
}

# 简易长参数解析
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gaussians)       GAUSSIANS="$2"; shift 2;;
    --img-paths)       IMG_PATHS="$2"; shift 2;;
    --outdir)          OUTDIR="$2"; shift 2;;

    --fov)             FOV="$2"; shift 2;;
    --imgw)            IMGW="$2"; shift 2;;
    --imgh)            IMGH="$2"; shift 2;;
    --near)            NEAR="$2"; shift 2;;
    --far)             FAR="$2"; shift 2;;

    --min-visible)     MIN_VISIBLE="$2"; shift 2;;
    --overlap-ratio)   OVERLAP_RATIO="$2"; shift 2;;
    --ds)              DS="$2"; shift 2;;

    --src)             SRC_HOST_DIR="$2"; shift 2;;
    --name)            CONTAINER_NAME="$2"; shift 2;;
    --image)           IMAGE="$2"; shift 2;;

    -h|--help)         print_help;;
    *) echo "Unknown option: $1"; print_help;;
  esac
done

# ========= 预检 =========
if [[ ! -d "$SRC_HOST_DIR" ]]; then
  echo "ERROR: SRC_HOST_DIR 不存在: $SRC_HOST_DIR"
  exit 1
fi

# 尝试为 X11 放行
if command -v xhost >/dev/null 2>&1; then
  xhost +local:root >/dev/null 2>&1 || true
fi

# ========= 组装要在容器中执行的命令 =========
CONTAINER_CMD="cd manual_splat && \
python3 manual_plane.py \
    --gaussians \"$GAUSSIANS\" \
    --fov $FOV \
    --imgw $IMGW \
    --imgh $IMGH \
    --near $NEAR \
    --far $FAR \
    --min_visible $MIN_VISIBLE \
    --overlap_ratio $OVERLAP_RATIO \
    --ds $DS \
    --img_paths \"$IMG_PATHS\" \
    --pose_outdir \"$OUTDIR\""

echo "==> 启动容器并自动运行 manual_plane.py ..."
echo "    镜像:       $IMAGE"
echo "    容器名:     $CONTAINER_NAME"
echo "    源码映射:   $SRC_HOST_DIR -> /workspace/src"
echo "    运行命令:   $CONTAINER_CMD"
echo

# ========= 运行 docker =========
docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e FORCE_CUDA=1 \
  -e QT_X11_NO_MITSHM=1 \
  -e DISPLAY="${DISPLAY:-}" \
  -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}" \
  -e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}" \
  -v "${SRC_HOST_DIR}":/workspace/src \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -w /workspace/src \
  "${IMAGE}" \
  bash -lc "$CONTAINER_CMD"
