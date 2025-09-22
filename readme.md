# 高斯溅射交互式路径规划与渲染工具
- [高斯溅射交互式路径规划与渲染工具](#高斯溅射交互式路径规划与渲染工具)
  - [功能一览](#功能一览)
  - [开发与部署](#开发与部署)
    - [开发模式（VS Code Dev Containers）](#开发模式vs-code-dev-containers)
    - [部署模式（最小运行镜像）](#部署模式最小运行镜像)
    - [脚本参数：`docker/run_all.bash`](#脚本参数dockerrun_allbash)
  - [启动交互式规划器](#启动交互式规划器)
    - [界面与交互](#界面与交互)
  - [致谢](#致谢)


本工具能够读取高斯溅射模型，设置水平切片平面并通过点击生成一个可被自动平滑并均匀采样的路径，在每个采样点上进行图像渲染并保存。

---

## 功能一览
- 载入高斯溅射模型（`.npz` 或 `.ply`，支持 3DGS 的 `f_dc_* / scale_* / rot_* / opacity` 等字段）
- 交互设置水平切片平面 `z = const`
- 在平面上**顺序点击**路径点（默认：**按住 Shift + 左键**添加点，避免与旋转冲突）
- 拟合 B 样条并按间距采样
- 依据“**最小可见点数** + **相邻帧可见点重叠**”筛选相机位姿
- 导出 `planned_poses.json`（含内参、`T_wc` 外参、统计信息）
- 调用渲染脚本批量出图

---

## 开发与部署

### 开发模式（VS Code Dev Containers）

前置：已安装 Docker、NVIDIA 驱动与 NVIDIA Container Toolkit（需要 GPU），VS Code 与 Dev Containers 插件。

步骤：
- 打开本仓库（文件夹根目录）。
- VS Code 命令面板运行“Dev Containers: Open Folder in Container…”。
- 首次会使用 `docker/Dockerfile.dev` 构建开发镜像，并通过 `docker-compose.yml` 将整个仓库挂载到容器的 `/workspace`。

进入容器后：
```bash
# 打开交互式规划器（需要 X11，Linux 主机需允许 X：xhost +local:root）
cd /workspace/src/manual_splat
bash run.bash
# 或直接运行脚本
# python3 manual_plane.py --gaussians /workspace/src/data/point_cloud_gs.ply --imgw 1920 --imgh 1080
```

说明：
- 开发容器使用 `docker-compose.yml` 中的 `dev` 服务，已设置 `gpus: all`、共享 X11 套接字等；工作目录为 `/workspace/src`。
- 宿主机代码实时挂载到容器，改代码无需重建镜像。

### 部署模式（最小运行镜像）

推荐使用脚本一键构建并运行（默认仅当镜像不存在时才构建）：
```bash
bash docker/run_all.bash
```
可选：
- 跳过构建：`bash docker/run_all.bash --no-build`
- 强制重建：`DO_BUILD=true bash docker/run_all.bash`
- 使用开发镜像（含编译工具链）：`bash docker/run_all.bash --dev`

也可以手动构建/运行部署镜像：
```bash
docker build -f docker/Dockerfile.deploy -t gsr:deploy .
docker run --rm -it --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/src/data/point_cloud_gs.ply:/mnt/gaussians.ply:ro \
  -v $(pwd)/output:/workspace/output \
  -w /workspace/src gsr:deploy bash
```

### 脚本参数：`docker/run_all.bash`

必备输入：建议显式传入高斯模型路径 `--gaussians`（默认路径可能不存在）。

通用：
- `--gaussians PATH`：高斯溅射模型（.ply / .npz），挂载到容器 `/mnt/gaussians.ply`。
- `--outdir DIR`：输出目录（图像与位姿 JSON），默认 `./output`。

相机参数：
- `--fov N`：视场（度），默认 70。
- `--imgw N` / `--imgh N`：图像宽/高，默认 1920×1080。
- `--near F` / `--far F`：近/远裁剪平面，默认 0.1 / 20.0。

采样与筛选：
- `--min-visible N`：每帧最小可见点数，默认 800。
- `--overlap-ratio F`：相邻帧可见点重叠（Jaccard），默认 0.4。
- `--ds F`：样条采样间距（米），默认 0.3。

容器参数：
- `--name NAME`：容器名，默认 `gsr`。
- `--image IMAGE`：镜像名:tag，默认 `gsr:local`。
- `--no-build`：跳过构建（默认自动检测镜像是否存在）。
- `--dockerfile FILE`：指定 Dockerfile（默认使用 `docker/Dockerfile.deploy`）。
- `--dev`：改用开发镜像（`docker/Dockerfile.dev`）。

环境变量：
- `DO_BUILD=auto|true|false`：构建策略（默认 auto：本地缺失才构建；true：总是构建；false：从不构建）。
- `DOCKERFILE=PATH`：覆盖 Dockerfile 路径。

示例：
```bash
# 用部署镜像，自动构建（缺失时），并渲染
bash docker/run_all.bash \
  --gaussians $(pwd)/examples/point_cloud_gs.ply \
  --outdir $(pwd)/output \
  --fov 75 --imgw 2560 --imgh 1440

# 使用开发镜像（包含编译工具链）并强制重建
DO_BUILD=true bash docker/run_all.bash --dev \
  --gaussians $(pwd)/examples/point_cloud_gs.ply \
  --outdir $(pwd)/output
```

---

## 启动交互式规划器
```bash
cd manual_splat
bash run.bash
```

**[Update]** 一键脚本（部署模式）
```bash
bash docker/run_all.bash   # 自动构建/启动
```

### 界面与交互

- **视角控制**：左键拖动旋转、右键拖动平移、滚轮缩放  

- **选平面**：右侧 `Slice plane z` 输入框设置 z，点 `Update / Show plane` 刷新蓝色平面 

  ![image-20250918155324574](./images/image-20250702170358302.png)

- **拾取路径点**：**按住 Shift + 左键**在平面上依次点击  
  
  - `Undo last point` 撤销最后一点  
  - `Clear points` 清空所有点  
  
- **拟合与采样**：`Fit spline & sample`

  ![image-20250918155435842](./images/image-20250918155435842.png)

- **生成与导出**：`Generate poses & export JSON & render images` → 输出json和png

  **[Update 2025/09/19]** 同时会生成每个位置对应的深度图，以16 位 PNG 图像按照mm(`depth_scale=1000.0`)为单位保存。

  ![image-20250918155820867](./images/image-20250918155820867.png)
  
  ![image-20250918160639251](./images/frame_0007.png)
  
  ![image-20250918160639251](./images/frame_0007_d.png)

---

## 致谢
- Open3D GUI 框架
- gsplat 渲染器
- Gaussian Splatting 原始论文 & 实现
