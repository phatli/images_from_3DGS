#!/usr/bin/env bash
# ===============================
# Run manual_plane.py with params
# ===============================

# 输入高斯模型文件
GAUSSIANS="/workspace/src/data/point_cloud_gs.ply"

# 输出图像保存目录
IMG_PATHS="/workspace/src/manual_splat/manual_out_img"

# 输出图像保存目录
OUTDIR="/workspace/src/manual_splat/manual_out_poses"

# 相机参数
FOV=70
IMGW=1920
IMGH=1080
NEAR=0.1
FAR=20.0

# 采样与筛选参数
MIN_VISIBLE=800
OVERLAP_RATIO=0.4
DS=0.3

# 启动 Python 脚本
python3 manual_plane.py \
  --gaussians "$GAUSSIANS" \
  --fov $FOV \
  --imgw $IMGW \
  --imgh $IMGH \
  --near $NEAR \
  --far $FAR \
  --min_visible $MIN_VISIBLE \
  --overlap_ratio $OVERLAP_RATIO \
  --ds $DS \
  --img_paths "$IMG_PATHS" \
  --pose_outdir "$OUTDIR"

