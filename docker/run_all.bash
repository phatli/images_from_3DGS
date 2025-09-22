#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONTAINER_NAME="${CONTAINER_NAME:-gsr}"
# 默认使用已发布镜像，只有显式指定本地构建时才从头构建
IMAGE="${IMAGE:-phatli/images_from_3dgs:latest}"
# 构建策略：false(默认，跳过)、true(强制构建)、auto(若无则构建)
DO_BUILD=${DO_BUILD:-false}
# 选择 Dockerfile：默认使用部署版
DOCKERFILE="${DOCKERFILE:-${ROOT_DIR}/docker/Dockerfile.deploy}"

GAUSSIANS="${GAUSSIANS:-${ROOT_DIR}/src/data/point_cloud_gs.ply}"
OUTDIR="${OUTDIR:-${ROOT_DIR}/output}"

FOV="${FOV:-70}"
IMGW="${IMGW:-1920}"
IMGH="${IMGH:-1080}"
NEAR="${NEAR:-0.1}"
FAR="${FAR:-20.0}"

MIN_VISIBLE="${MIN_VISIBLE:-800}"
OVERLAP_RATIO="${OVERLAP_RATIO:-0.4}"
DS="${DS:-0.3}"

print_help() {
  cat <<EOF
Usage: $(basename "$0") [options]

# 路径（Host 上路径，脚本自动做挂载）
  --gaussians PATH           默认(Host): $GAUSSIANS (单个文件)
  --outdir DIR               默认(Host): $OUTDIR (单一输出目录，容器内用于图像与位姿)

# 相机参数（与原版保持一致）
  --fov N                    默认: $FOV
  --imgw N                   默认: $IMGW
  --imgh N                   默认: $IMGH
  --near F                   默认: $NEAR
  --far F                    默认: $FAR

# 采样与筛选（与原版保持一致）
  --min-visible N            默认: $MIN_VISIBLE
  --overlap-ratio F          默认: $OVERLAP_RATIO
  --ds F                     默认: $DS

# Docker 相关
  --name NAME                容器名，默认: $CONTAINER_NAME
  --image IMAGE              镜像名:tag（本地构建），默认: $IMAGE
  --build-local              显式本地构建（默认跳过构建，使用已发布镜像）
  --no-build                 跳过构建镜像（与默认一致）
  --dockerfile FILE          指定 Dockerfile，默认: $DOCKERFILE
  --dev                      使用开发版 Dockerfile (docker/Dockerfile.dev)

示例：
  $(basename "$0") \
    --gaussians ${ROOT_DIR}/src/data/xyz.ply \
    --outdir ${ROOT_DIR}/output \
    --fov 75 --imgw 2560 --imgh 1440
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gaussians)       GAUSSIANS="$2"; shift 2;;
    --outdir)          OUTDIR="$2"; shift 2;;

    --fov)             FOV="$2"; shift 2;;
    --imgw)            IMGW="$2"; shift 2;;
    --imgh)            IMGH="$2"; shift 2;;
    --near)            NEAR="$2"; shift 2;;
    --far)             FAR="$2"; shift 2;;

    --min-visible)     MIN_VISIBLE="$2"; shift 2;;
    --overlap-ratio)   OVERLAP_RATIO="$2"; shift 2;;
    --ds)              DS="$2"; shift 2;;

    --name)            CONTAINER_NAME="$2"; shift 2;;
    --image)           IMAGE="$2"; shift 2;;
    --dockerfile)      DOCKERFILE="$2"; shift 2;;
    --dev)             DOCKERFILE="${ROOT_DIR}/docker/Dockerfile.dev"; shift 1;;
    --build-local)     DO_BUILD=true; shift 1;;
    --no-build)        DO_BUILD=false; shift 1;;

    -h|--help)         print_help;;
    *) echo "Unknown option: $1"; print_help;;
  esac
done

abs_path() { (
  cd "$(dirname "$1")" >/dev/null 2>&1 && \
  printf "%s/%s" "$(pwd -P)" "$(basename "$1")"
) }

GAUSSIANS_HOST_ABS="$(abs_path "$GAUSSIANS")"
OUTDIR_HOST_ABS="$(abs_path "$OUTDIR")"

if [[ ! -f "$GAUSSIANS_HOST_ABS" ]]; then
  echo "ERROR: gaussians 文件不存在: $GAUSSIANS_HOST_ABS"; exit 1
fi
mkdir -p "$OUTDIR_HOST_ABS"

if command -v xhost >/dev/null 2>&1; then
  xhost +local:root >/dev/null 2>&1 || true
fi

GAUSSIANS_C="/mnt/gaussians.ply"
OUTDIR_C="/workspace/output"

CONTAINER_CMD="cd manual_splat && \
python3 manual_plane.py \
    --gaussians \"$GAUSSIANS_C\" \
    --fov $FOV \
    --imgw $IMGW \
    --imgh $IMGH \
    --near $NEAR \
    --far $FAR \
    --min_visible $MIN_VISIBLE \
    --overlap_ratio $OVERLAP_RATIO \
    --ds $DS \
    --img_paths \"$OUTDIR_C\" \
    --pose_outdir \"$OUTDIR_C\""

echo "==> 使用 Dockerfile: ${DOCKERFILE}"
if [[ "${DO_BUILD}" == "false" ]]; then
  echo "==> 跳过构建镜像，使用已有镜像: ${IMAGE}"
elif [[ "${DO_BUILD}" == "true" ]]; then
  echo "==> 强制构建镜像: ${IMAGE} (Dockerfile: ${DOCKERFILE})"
  docker build -f "${DOCKERFILE}" -t "${IMAGE}" "${ROOT_DIR}"
else
  if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "==> 发现已有镜像: ${IMAGE}，跳过构建"
  else
    echo "==> 本地未找到镜像，开始构建: ${IMAGE} (Dockerfile: ${DOCKERFILE})"
    docker build -f "${DOCKERFILE}" -t "${IMAGE}" "${ROOT_DIR}"
  fi
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "移除已存在容器: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

echo "==> 启动容器并自动运行 manual_plane.py ..."
echo "    镜像:       $IMAGE"
echo "    容器名:     $CONTAINER_NAME"
echo "    Gaussians:  $GAUSSIANS_HOST_ABS -> $GAUSSIANS_C"
echo "    Outdir:     $OUTDIR_HOST_ABS -> $OUTDIR_C"

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
  -v "${GAUSSIANS_HOST_ABS}:${GAUSSIANS_C}:ro" \
  -v "${OUTDIR_HOST_ABS}:${OUTDIR_C}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -w /workspace/src \
  "${IMAGE}" \
  bash -lc "$CONTAINER_CMD"
