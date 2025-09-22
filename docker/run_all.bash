#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONTAINER_NAME="${CONTAINER_NAME:-gsr}"
# 默认使用已发布镜像，只有显式指定本地构建时才从头构建
IMAGE="${IMAGE:-phatli/images_from_3dgs:latest}"
# 构建策略：false(默认，跳过)、true(强制构建)、auto(若无则构建)
DO_BUILD=${DO_BUILD:-false}
# 是否强制要求在容器内检测到 CUDA（默认 true）；否则将中止以避免 CPU 回退造成“卡死”
REQUIRE_GPU=${REQUIRE_GPU:-true}
# 是否使用 --runtime=nvidia（在 dind/非默认 runtime 下通常需要）。默认自动：预检失败时回退尝试。
RUNTIME_ARG=${RUNTIME_ARG:-}
# IPC 与资源限制（默认避免 --ipc=host，使用较大的 /dev/shm）
IPC_ARG=${IPC_ARG:---shm-size=8g}
CPUS_ARG=${CPUS_ARG:-}
MEM_ARG=${MEM_ARG:-}
OMP_THREADS=${OMP_THREADS:-2}
USE_X11=${USE_X11:-true}
PYTORCH_CUDA_ALLOC_CONF_ENV=${PYTORCH_CUDA_ALLOC_CONF_ENV:-max_split_size_mb:64}
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
  --allow-cpu                允许在未检测到 GPU 时继续运行（默认不允许）
  --build-local              显式本地构建（默认跳过构建，使用已发布镜像）
  --no-build                 跳过构建镜像（与默认一致）
  --dockerfile FILE          指定 Dockerfile，默认: $DOCKERFILE
  --dev                      使用开发版 Dockerfile (docker/Dockerfile.dev)
  --runtime-nvidia           强制在 docker run 时添加 --runtime=nvidia（dind 常用）
  --ipc-host                 使用 --ipc=host（默认使用独立 IPC 并设置 --shm-size=8g）
  --cpus N                   限制容器可用 CPU 数（例如 4）
  --mem SIZE                 限制内存（例如 16g）；同时限制 swap 为同值
  --omp N                    设定数学库线程数（默认 2）
  --safe                     降低渲染负载（分辨率 1280x720，ds=0.5）
  --no-x                     不挂载 X/Wayland（无 GUI；手动用途）

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
    --allow-cpu)       REQUIRE_GPU=false; shift 1;;
    --runtime-nvidia)  RUNTIME_ARG="--runtime=nvidia"; shift 1;;
    --ipc-host)        IPC_ARG="--ipc=host"; shift 1;;
    --cpus)            CPUS_ARG="--cpus $2"; shift 2;;
    --mem)             MEM_ARG="--memory $2 --memory-swap $2"; shift 2;;
    --omp)             OMP_THREADS="$2"; shift 2;;
    --safe)            IMGW=1280; IMGH=720; DS=0.5; shift 1;;
    --no-x)            USE_X11=false; shift 1;;

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
mkdir -p "$OUTDIR_HOST_ABS" "$OUTDIR_HOST_ABS/rgb" "$OUTDIR_HOST_ABS/depth" "$OUTDIR_HOST_ABS/poses"

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
    --img_paths \"$OUTDIR_C/rgb\" \
    --depth_paths \"$OUTDIR_C/depth\" \
    --pose_outdir \"$OUTDIR_C/poses\""

# --------------------
# GPU 预检（防止 CPU 回退）
# --------------------
gpu_sanity_check() {
  # 将测试脚本写到宿主的临时文件并挂载进去，避免嵌套 here-doc 造成的转义问题
  local tmp_py
  tmp_py="$(mktemp)"
  cat >"${tmp_py}" <<'PY'
import sys, torch

try:
    import gsplat as gs
    gs_ok = True
except Exception as e:
    gs_ok = False
    err = str(e)

print("torch.cuda.is_available=", torch.cuda.is_available())
print("torch.version.cuda=", getattr(torch.version, "cuda", None))
print("device_count=", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name=", torch.cuda.get_device_name(0))
print("gsplat_import=", gs_ok)

# 返回 0 仅当 GPU 可用且 gsplat 可导入
sys.exit(0 if (torch.cuda.is_available() and gs_ok) else 2)
PY

  echo "==> GPU 预检: 启动一次性容器检测 CUDA/gsplat ..."
  if docker run --rm \
      ${RUNTIME_ARG:+$RUNTIME_ARG} \
      --gpus all \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
      -w /workspace/src \
      -v "${tmp_py}:/tmp/gpu_check.py:ro" \
      "${IMAGE}" \
      bash -lc "python3 /tmp/gpu_check.py"; then
    echo "==> GPU 就绪（容器内已检测到 CUDA 且可导入 gsplat）"
    rm -f "${tmp_py}"
    return 0
  else
    # 如果未强制 runtime，尝试一次 --runtime=nvidia（适配 dind/非默认 runtime）
    if [[ -z "${RUNTIME_ARG}" ]]; then
      echo "==> 重试：使用 --runtime=nvidia ..."
      if docker run --rm \
          --runtime=nvidia --gpus all \
          -e NVIDIA_VISIBLE_DEVICES=all \
          -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
          -w /workspace/src \
          -v "${tmp_py}:/tmp/gpu_check.py:ro" \
          "${IMAGE}" \
          bash -lc "python3 /tmp/gpu_check.py"; then
        echo "==> GPU 就绪（通过 --runtime=nvidia）。将对后续运行启用该参数。"
        RUNTIME_ARG="--runtime=nvidia"
        rm -f "${tmp_py}"
        return 0
      fi
    fi
    echo "!! 未检测到 GPU 或 gsplat 异常（可能在 dind/或未启用 nvidia runtime）。" >&2
    rm -f "${tmp_py}"
    return 2
  fi
}

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

# 若需要 GPU 且检测失败则直接退出，避免 CPU 回退导致系统负载过高
if [[ "${REQUIRE_GPU}" == "true" ]]; then
  if ! gpu_sanity_check; then
    echo "==> 中止：未检测到可用 GPU。\n提示：\n - 避免 dind，改为在宿主 Docker 引擎上直接运行（确保安装 NVIDIA Container Toolkit）。\n - 或显式允许 CPU 回退：添加 --allow-cpu（可能极慢/高负载）。" >&2
    exit 2
  fi
else
  echo "==> 已允许 CPU 回退（--allow-cpu）；可能非常缓慢并造成高负载。"
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

# 构建 docker run 参数，便于按需增删
RUN_ARGS=(
  --rm -it
  ${RUNTIME_ARG:+$RUNTIME_ARG}
  --name "${CONTAINER_NAME}"
  --gpus all
  ${IPC_ARG}
)

if [[ -n "${CPUS_ARG}" ]]; then
  RUN_ARGS+=( ${CPUS_ARG} )
fi
if [[ -n "${MEM_ARG}" ]]; then
  RUN_ARGS+=( ${MEM_ARG} )
fi

# 通用环境变量
RUN_ARGS+=(
  -e NVIDIA_VISIBLE_DEVICES=all
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
  -e FORCE_CUDA=1
  -e OMP_NUM_THREADS="${OMP_THREADS}"
  -e OPENBLAS_NUM_THREADS="${OMP_THREADS}"
  -e MKL_NUM_THREADS="${OMP_THREADS}"
  -e NUMEXPR_NUM_THREADS="${OMP_THREADS}"
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_ENV}"
)

# X11/Wayland（可选）
if [[ "${USE_X11}" == "true" ]]; then
  RUN_ARGS+=(
    -e QT_X11_NO_MITSHM=1
    -e DISPLAY="${DISPLAY:-}"
    -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}"
    -e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}"
    -v /tmp/.X11-unix:/tmp/.X11-unix
  )
fi

# 挂载数据与工作目录
RUN_ARGS+=(
  -v "${GAUSSIANS_HOST_ABS}:${GAUSSIANS_C}:ro"
  -v "${OUTDIR_HOST_ABS}:${OUTDIR_C}"
  -w /workspace/src
)

docker run "${RUN_ARGS[@]}" \
  "${IMAGE}" \
  bash -lc "$CONTAINER_CMD"
