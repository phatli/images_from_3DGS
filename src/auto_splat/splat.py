# pip install gsplat numpy scipy imageio transforms3d
import os, math, json
import numpy as np
import imageio.v2 as imageio

from transforms3d.quaternions import mat2quat
from transforms3d.euler import euler2mat

import torch
import gsplat as gs
import yaml

from plyfile import PlyData
from load_ply import load_gaussians_from_ply
from skeleton_polyline import make_skeleton_polyline, pose_from_tangent, step_from_overlap,  density_roi, visible_count
from free_space_path import (fit_principal_plane, gaussian_radius_from_scale,
                             build_2d_occupancy_and_distance,
                             extract_closed_loop_from_distance,
                             resample_smooth_closed_curve,
                             lift_loop_and_orient)
from vis import viz_open3d, visualize_path_open3d, visualize_occ_plane_and_points_open3d

def refine_cam_by_shifting(
    loop_xyz, idx_hint, cam_init, shift_step, tries,
    world_pts_np,
    means_t, K_1x33, W, H, DEVICE,
    d_min, d_max, d_target, min_vis, vis_sample_max
):
    """
    围绕初始位置，沿切线方向 ±shift_step 反复微移（tries 次），每次试 orientation 集合。
    返回 (R,t,cam_final,vis,z,ok)
    """
    M = len(loop_xyz)
    i = int(idx_hint) % M
    def interp_on_edge(i, alpha):
        a = loop_xyz[i]
        b = loop_xyz[(i+1)%M]
        return a + alpha*(b-a)

    # 初始落点的边和alpha
    a = loop_xyz[i]; b = loop_xyz[(i+1)%M]
    seg = np.linalg.norm(b-a)
    if seg < 1e-9:
        return None, None, cam_init, 0, None, False
    # 反推 alpha
    alpha = np.clip(np.dot(cam_init - a, (b-a))/ (seg*seg), 0.0, 1.0)

    # 尝试序列：0, +1, -1, +2, -2, ...
    order = [0]
    for k in range(1, tries+1):
        order += [k, -k]

    for k in order:
        # 计算新的 (edge, alpha)；当越界时进位到下一段
        delta = k * (shift_step / max(1e-6, seg))
        j = i
        al = alpha + delta
        # 规范化到 [0,1) 并更新段索引
        while al >= 1.0:
            al -= 1.0
            j = (j+1) % M
        while al < 0.0:
            j = (j-1+M) % M
            al += 1.0
        cam = interp_on_edge(j, al)
        prev_pt = loop_xyz[(j-1+M)%M]
        next_pt = loop_xyz[(j+1)%M]

        R, t, vis, z, ok = choose_orientation_for_cam(
            cam, prev_pt, next_pt,
            world_pts_np,
            means_t, K_1x33, W, H, DEVICE,
            d_min, d_max, d_target,
            min_vis, vis_sample_max
        )
        if ok:
            return R, t, cam, vis, z, True

    # 仍不达标，返回最佳（由 choose 中的打分给出）
    # 为简洁，这里最后再跑一次 choose，用 cam_init
    R, t, vis, z, ok = choose_orientation_for_cam(
        cam_init, loop_xyz[(i-1+M)%M], loop_xyz[(i+1)%M],
        world_pts_np,
        means_t, K_1x33, W, H, DEVICE,
        d_min, d_max, d_target,
        min_vis, vis_sample_max
    )
    return R, t, cam_init, vis, z, ok

def pose_look_at(cam, target, up=np.array([0,0,1], np.float32)):
    z = (target - cam); z /= (np.linalg.norm(z)+1e-9)
    x = np.cross(up, z); x /= (np.linalg.norm(x)+1e-9)
    y = np.cross(z, x)
    R = np.stack([x,y,z], axis=0).astype(np.float32)
    t = -R @ cam.astype(np.float32)
    return R, t

def local_centroid(world_pts, center, radius=2.0, fallback_k=8000):
    if world_pts.shape[0]==0: return center
    d = np.linalg.norm(world_pts - center[None,:], axis=1)
    m = d <= radius
    if np.any(m):
        return world_pts[m].mean(axis=0)
    k = min(fallback_k, world_pts.shape[0])
    idx = np.argpartition(d, k)[:k]
    return world_pts[idx].mean(axis=0)

def choose_orientation_for_cam(
    cam, prev_pt, next_pt,
    world_pts_np,
    means_t, K_1x33, W, H, DEVICE,
    d_min, d_max, d_target,
    min_vis, vis_sample_max,
    beta=0.5
):
    """
    返回(best_R, best_t, best_vis, best_z, ok)
    ok=True 表示满足 min_vis & z ∈ [d_min,d_max]
    """
    tdir = next_pt - prev_pt
    if np.linalg.norm(tdir) < 1e-9:
        tdir = np.array([1,0,0], np.float32)
    tdir = tdir / (np.linalg.norm(tdir)+1e-9)

    glob_centroid = world_pts_np.mean(axis=0)
    loc_centroid  = local_centroid(world_pts_np, cam, radius=max(d_max*1.2, 1.0))

    candidates = []
    # 1) 纯切线（相机前方一定距离的目标点）
    candidates.append(cam + tdir)
    # 2) 全局质心
    candidates.append(glob_centroid)
    # 3) 局部质心
    candidates.append(loc_centroid)
    # 4-5) 混合（偏切线/偏中心）
    candidates.append(0.7*(cam+tdir) + 0.3*loc_centroid)
    candidates.append(0.4*(cam+tdir) + 0.6*loc_centroid)

    best = None
    for tgt in candidates:
        R, t = pose_look_at(cam, tgt)
        T = np.eye(4, dtype=np.float32); T[:3,:3]=R; T[:3,3]=t
        T_1x44 = torch.tensor(T, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        vis, z_med = visible_count(means_t, K_1x33, T_1x44, W, H, sample_N=vis_sample_max)
        sc = -1e18
        if z_med is not None:
            sc = vis - beta * (abs(z_med - d_target) / max(1e-6, d_target)) * float(min_vis)
        cand = (sc, R, t, vis, z_med)
        if (best is None) or (sc > best[0]): best = cand

    _, Rb, tb, visb, zb = best
    ok = (visb >= min_vis) and (zb is not None) and (d_min <= zb <= d_max)
    return Rb, tb, visb, zb, ok

def polyline_arclen(P):
    d = np.linalg.norm(np.diff(P, axis=0, append=P[:1]), axis=1)  # 闭环
    s = np.concatenate([[0], np.cumsum(d)])  # 长度数组(含首0，末为总长)
    return s

def sample_positions_along_loop(loop_xyz, step_len):
    """
    闭环按弧长等距采样；返回(positions, indices_hint)
    indices_hint 给出每个样本大致落在哪段，便于取切线。
    """
    P = loop_xyz
    seg = np.linalg.norm(np.diff(P, axis=0, append=P[:1]), axis=1)
    L = seg.sum()
    if L < 1e-6:
        raise RuntimeError("闭环长度太短")
    n = max(1, int(np.floor(L / max(1e-6, step_len))))
    targets = np.linspace(0, L, n, endpoint=False)
    pos = []
    idx_hint = []
    acc = 0.0
    i = 0
    for t in targets:
        while acc + seg[i] < t:
            acc += seg[i]; i = (i+1) % len(P)
        ratio = (t - acc) / max(1e-9, seg[i])
        a = P[i]
        b = P[(i+1)%len(P)]
        pos.append(a + ratio*(b-a))
        idx_hint.append(i)
    return np.asarray(pos, dtype=np.float32), np.asarray(idx_hint, dtype=np.int32)

def step_from_overlap_static(hfov_deg, overlap, z_target):
    HFOV = math.radians(hfov_deg)
    return 2.0 * z_target * math.tan(HFOV/2.0) * (1.0 - overlap)

def lift_loop_positions(loop_uv, plane_to_world, plane_height_offset):
    """仅把闭环2D曲线抬回3D，返回 loop_xyz(M,3)"""
    M = loop_uv.shape[0]
    loop_p = np.zeros((M,3), dtype=np.float32)
    loop_p[:,0:2] = loop_uv
    loop_p[:,2]   = float(plane_height_offset)
    loop_xyz = plane_to_world(loop_p)
    return loop_xyz

# ---------------- 参数区 ----------------
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# # 使用配置
W = cfg["render"]["width"]
H = cfg["render"]["height"]
HFOV_deg = cfg["render"]["hfov_deg"]
OVERLAP = cfg["render"]["overlap"]
FPS = cfg["render"]["fps"]

voxel = cfg["roi"]["voxel"]
min_count = cfg["roi"]["min_count"]
keep_top_percent = cfg["roi"]["keep_top_percent"]

PLY_PATH = cfg["paths"]["ply_path"]
OUT_DIR = cfg["paths"]["out_dir"]
DEVICE = cfg["device"]

SCALE_GAIN = cfg["render"]["scale_gain"]

os.makedirs(f"{OUT_DIR}/images", exist_ok=True)
# print(f"配置读取成功: W={W}, H={H}, HFOV={HFOV_deg}, ply={PLY_PATH}")

# 计算焦距
HFOV = math.radians(HFOV_deg)
fx = fy = W / (2*math.tan(HFOV/2))
cx, cy = W/2.0, H/2.0

# ---------------- 加载3DGS ----------------

# scene = gs.GaussianScene.from_ply(PLY_PATH, device=DEVICE)
scene = load_gaussians_from_ply(PLY_PATH, device=DEVICE)
# xyz_np = scene['xyz'].cpu().numpy()

# ---- 计算基于密度的 ROI 再计算bounding box ----
xyz_np = scene['xyz'].cpu().numpy()
bbox_min_all, bbox_max_all = xyz_np.min(axis=0), xyz_np.max(axis=0)  # 全局 bbox
bbox_min, bbox_max = density_roi(
    xyz_np, voxel = voxel,
    min_count = min_count,
    keep_top_percent = keep_top_percent
)

# polyline = make_skeleton_polyline(
#     xyz_np, bbox_min, bbox_max,
#     voxel_track=cfg["skeleton"]["voxel_track"],
#     min_component_ratio=cfg["skeleton"]["min_component_ratio"],
#     smooth_window=cfg["skeleton"]["smooth_window"],
#     progress="tqdm"   # ← 改成 "live" 可看实时折线增长
# )

# if polyline.shape[0] < 2:
#     raise RuntimeError("骨架折线点过少，调大 roi.voxel 或降低 roi.min_count")

# center = 0.5*(bbox_min + bbox_max)
# S = float(np.linalg.norm((bbox_max - bbox_min)))

# bbox_min, bbox_max = xyz_np.min(axis=0), xyz_np.max(axis=0)
# center = 0.5*(bbox_min + bbox_max)
# S = float(np.linalg.norm((bbox_max - bbox_min)))

# ---- 相机内参张量（批大小=1）----
K_torch = torch.tensor([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]], dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,3,3)

# ---- 按 gsplat 需要的形状准备张量 ----
# load_gaussians_from_ply 直接从建好的3d gsplat ply文件读取到了 scale/rot/opacity/rgb
# 必要键：xyz(N,3), scale(N,3), rot(N,4=四元数), opacity(N,1), rgb(N,3) 
# ---- 按 gsplat 需要的形状准备张量（不加 batch 维）----
for k in scene:
    scene[k] = scene[k].to(DEVICE).contiguous().float()

means  = scene['xyz']                    # (N,3)  
colors = scene['rgb']                    # (N,3)  
scales = scene['scale']                  # (N,3) 

# opacity 是 logit，需要激活到 [0,1]
opacs  = torch.sigmoid(scene['opacity'].squeeze(-1))   

# 3DGS 的 scale_* 是 log 空间，渲染前要 exp 成线性尺寸
# log->linear，并给个可调增益
scales = torch.exp(scene['scale']) * SCALE_GAIN  # (N,3)

# 四元数顺序：多数导出是 xyzw，gsplat 要 wxyz
q = scene['rot']                        # (N,4)
q = q / (q.norm(dim=-1, keepdim=True) + 1e-9)  # 规范化，避免数值飘
q_wxyz = torch.stack([q[:,3], q[:,0], q[:,1], q[:,2]], dim=-1)  # (N,4)

print("means xyz min/max", means.min().item(), means.max().item())
print("scales (raw) min/max", scene['scale'].min().item(), scene['scale'].max().item())
print("scales exp min/max", scales.min().item(), scales.max().item())
print("opacs sigmoid min/max", opacs.min().item(), opacs.max().item())


# ==== 输入bbox_min, bbox_max====
# 可选：给 bbox 加一点 padding，避免边缘被截断
pad = float(cfg.get("bbox_pad", 0.0))  # config 可加这一项；没有则 0
bbox_min = np.asarray(bbox_min, dtype=np.float32) - pad
bbox_max = np.asarray(bbox_max, dtype=np.float32) + pad

# ---- 在 bbox 内筛选高斯 ----
xyz_all = scene['xyz']                     # torch (N,3)
mask_box = (xyz_all[:,0] >= bbox_min[0]) & (xyz_all[:,0] <= bbox_max[0]) & \
           (xyz_all[:,1] >= bbox_min[1]) & (xyz_all[:,1] <= bbox_max[1]) & \
           (xyz_all[:,2] >= bbox_min[2]) & (xyz_all[:,2] <= bbox_max[2])

# 若 bbox 太小导致一个点都没有，直接报错
if not torch.any(mask_box):
    raise RuntimeError("bbox 内没有高斯点，请检查 bbox_min/bbox_max 或调大 bbox_pad。")

# 只保留 bbox 内的高斯（用于“平面 + 规划 + 可见性”）
scene_box = {}
for k in scene:
    v = scene[k]
    if isinstance(v, torch.Tensor) and v.shape[0] == xyz_all.shape[0]:
        scene_box[k] = v[mask_box].contiguous()
    else:
        scene_box[k] = v  # 其他保持原样（比如常量）

# 重新绑定用于渲染/可见性/规划的张量（都用 bbox 里的）
means  = scene_box['xyz']                                    # (N_in,3)
colors = scene_box.get('rgb', None)
scales = torch.exp(scene_box['scale']) * SCALE_GAIN          # (N_in,3) 线性尺寸
opacs  = torch.sigmoid(scene_box['opacity'].squeeze(-1))     # (N_in,)

q = scene_box['rot']
q = q / (q.norm(dim=-1, keepdim=True) + 1e-9)
q_wxyz = torch.stack([q[:,3], q[:,0], q[:,1], q[:,2]], dim=-1)

# numpy 版（用于几何分析）
xyz_np = means.detach().cpu().numpy()

# ---------------- 轨迹采样工具 ----------------
def look_at(cam, target, up=np.array([0,0,1.0])):
    """
    返回 R(3x3), t(3,)
    cam/target: np.array(3,)
    """
    # z = (cam - target); z = z/np.linalg.norm(z)
    z = (target - cam); z = z/np.linalg.norm(z)
    x = np.cross(up, z); x = x/np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)  # world->cam
    t = -R @ cam
    return R, t

def colmap_qt_from_Rt(R, t):
    # COLMAP的images.txt使用世界到相机的四元数(qw qx qy qz)及平移t
    qx, qy, qz, qw = mat2quat(R.T)  # transforms3d返回的是 [w,x,y,z] for R->quat? 视实现调整
    # 注意 transforms3d.mat2quat(R) 返回 [w,x,y,z] for R (active)；根据你R定义确认是否需要转置/修正。
    qw, qx, qy, qz = float(qw), float(qx), float(qy), float(qz)
    return qw, qx, qy, qz, float(t[0]), float(t[1]), float(t[2])

# 由重叠推角步/平移步
def angle_step_from_overlap(hfov_deg, overlap):
    return math.radians(hfov_deg*(1.0 - overlap))*0.5  # 保守取半

def trans_step_from_overlap(z, hfov_deg, overlap):
    HFOV = math.radians(hfov_deg)
    return 2*z*math.tan(HFOV/2)*(1.0 - overlap)

# ---------------- 轨迹生成 ----------------

poses = []
img_paths = []

# ---------------- 尝试1：生成形状固定的“环绕+蛇形”轨迹 ----------------

# # 1) 环绕一圈
# radius = 0.3*S # 半径
# base_height = center[2] + 0.1*S
# dphi = angle_step_from_overlap(HFOV_deg, OVERLAP)  # 弧度
# num_orbit = max(24, int(2*math.pi/dphi))

# for i in range(num_orbit):
#     phi = i*(2*math.pi/num_orbit)
#     cam = np.array([
#         center[0] + radius*math.cos(phi),
#         center[1] + radius*math.sin(phi),
#         base_height
#     ])
#     R, t = look_at(cam, center)
#     poses.append((R, t))
#     img_paths.append(f"images/orbit_{i:04d}.png")

# # 2) 蛇形扫掠（两层高度）
# layers = [center[2]-0.05*S, center[2]+0.05*S]
# x0, x1 = bbox_min[0]-0.2*S, bbox_max[0]+0.2*S
# y0, y1 = bbox_min[1]-0.2*S, bbox_max[1]+0.2*S
# avg_z = radius
# step = trans_step_from_overlap(avg_z, HFOV_deg, OVERLAP)
# num_strips = int((y1-y0)/step)+1
# num_x = int((x1-x0)/step)+1

# for h in layers:
#     for sidx in range(num_strips):
#         yy = y0 + sidx*step
#         xs = np.linspace(x0, x1, num_x) if (sidx%2==0) else np.linspace(x1, x0, num_x)
#         for j, xx in enumerate(xs):
#             cam = np.array([xx, yy, h])
#             R, t = look_at(cam, center)
#             poses.append((R, t))
#             img_paths.append(f"images/sweep_{h:.2f}_{sidx:03d}_{j:03d}.png")

# ---------------- 尝试2：“骨架”+蛇形路径方案，依据GS模型动态生成轨迹 ----------------

# # 起点：折线起点
# p0 = polyline[0]
# p1 = polyline[1]
# tangent = (p1 - p0)
# R, t = pose_from_tangent(p0, tangent)

# # 可见性检查
# T = np.eye(4, dtype=np.float32); T[:3,:3]=R; T[:3,3]=t
# T_1x44 = torch.tensor(T, dtype=torch.float32, device=DEVICE).unsqueeze(0)
# vis, z_med = visible_count(means, K_torch, T_1x44, W, H, sample_N=cfg["sampling"]["vis_sample_max"])
# if vis >= cfg["sampling"]["min_visible"] and z_med is not None:
#     poses.append((R, t))
#     img_paths.append(f"images/adapt_0000.png")
#     start_idx = 0
# else:
#     # 向前推进直到找到第一个达标点
#     found=False
#     for i in range(1, min(50, polyline.shape[0]-1)):
#         p0 = polyline[i]
#         p1 = polyline[i+1]
#         tangent = (p1 - p0)
#         R, t = pose_from_tangent(p0, tangent)
#         T[:3,:3]=R; T[:3,3]=t
#         T_1x44 = torch.tensor(T, dtype=torch.float32, device=DEVICE).unsqueeze(0)
#         vis, z_med = visible_count(means, K_torch, T_1x44, W, H, sample_N=cfg["sampling"]["vis_sample_max"])
#         if vis >= cfg["sampling"]["min_visible"] and z_med is not None:
#             poses.append((R, t)); img_paths.append(f"images/adapt_{i:04d}.png")
#             found=True; start_idx=i; break
#     if not found:
#         raise RuntimeError("起点附近可见点过少，请调小 roi、降低 min_visible 或靠近密集区域。")

# # 沿折线推进
# i = start_idx
# skip_cnt = 0
# frame_idx = 1
# cur_pos = polyline[i]
# cur_R, cur_t = poses[-1]

# cur_zmed = z_med
# while i < polyline.shape[0]-1:
#     # 动态步长，满足 overlap
#     s = step_from_overlap(hfov_deg=cfg["render"]["hfov_deg"],
#                           overlap=cfg["render"]["overlap"],
#                           z_med=cur_zmed if cur_zmed is not None else np.linalg.norm(polyline[i+1]-polyline[i]),
#                           safety=cfg["sampling"]["step_safety"])

#     # 沿折线累计弧长 s，找到下一个目标点
#     acc = 0.0
#     j = i+1
#     while j < polyline.shape[0]:
#         seg = np.linalg.norm(polyline[j] - polyline[j-1])
#         if acc + seg >= s:
#             ratio = max(1e-6, (s - acc) / seg)
#             cand = polyline[j-1] + ratio * (polyline[j] - polyline[j-1])
#             break
#         acc += seg; j += 1
#     if j >= polyline.shape[0]:visualize_occ_plane_and_points_open3d
#         break
#     # 候选姿态
#     tangent = polyline[min(j+1, polyline.shape[0]-1)] - polyline[max(j-1,0)]
#     R, t = pose_from_tangent(cand, tangent)
#     T[:3,:3]=R; T[:3,3]=t
#     T_1x44 = torch.tensor(T, dtype=torch.float32, device=DEVICE).unsqueeze(0)
#     vis, z_med = visible_count(means, K_torch, T_1x44, W, H, sample_N=cfg["sampling"]["vis_sample_max"])

#     if vis >= cfg["sampling"]["min_visible"] and z_med is not None:
#         poses.append((R, t))
#         img_paths.append(f"images/adapt_{frame_idx:04d}.png")
#         frame_idx += 1
#         cur_zmed = z_med
#         i = j
#         skip_cnt = 0
#     else:
#         # 不达标就稍微向前滑动，直到达标或放弃
#         skip_cnt += 1
#         if skip_cnt > cfg["sampling"]["max_skip"]:
#             i = j
#             skip_cnt = 0
#         else:
#             i = j  # 继续往前尝试

# ==================== 尝试3：自由空间闭环路径 ====================

# # 1) 主平面 & 投影
# xyz_np = scene['xyz'].detach().cpu().numpy()   # (N,3)
# mu, Rpw, world_to_plane, plane_to_world, n_plane = fit_principal_plane(xyz_np)
# Xp = world_to_plane(xyz_np)                    # (N,3) 平面系

# # 2) 高斯占据半径（用 exp(scale)）
# scales_np = scales.detach().cpu().numpy()
# r_np = gaussian_radius_from_scale(scales_np, gain=cfg["free_space"]["radius_gain"])

# # 3) 建 2D 占据 + 距离场（用 u,v 两维；忽略平面系z）
# uvr = np.stack([Xp[:,0], Xp[:,1], r_np], axis=1)
# occ, D, origin_uv, (H2, W2) = build_2d_occupancy_and_distance(
#     uvr,
#     grid_res=cfg["free_space"]["grid_res"],
#     margin=4
# )

# 1) 主平面 & 投影（只用 bbox 内的点）
mu, Rpw, world_to_plane, plane_to_world, n_plane = fit_principal_plane(xyz_np)
Xp = world_to_plane(xyz_np)   # (N_in,3) 平面系

# 2) 高斯占据半径（只用 bbox 内的 scale）
scales_np = scales.detach().cpu().numpy()
r_np = gaussian_radius_from_scale(scales_np, gain=cfg["free_space"]["radius_gain"])

# 3) 建 2D 占据 + 距离场（用 u,v 两维；忽略平面系 z）
uvr = np.stack([Xp[:,0], Xp[:,1], r_np], axis=1)
occ, D, origin_uv, (H2, W2) = build_2d_occupancy_and_distance(
    uvr,
    grid_res=cfg["free_space"]["grid_res"],
    margin=4
)

visualize_occ_plane_and_points_open3d(
    occ, D, origin_uv, cfg["free_space"]["grid_res"],
    plane_to_world=plane_to_world,
    scene=scene_box,
    show_occupied=True,
    occ_alpha=0.35,
    downsample=2,                      # 可调：2/3 以加速
    mode="window",                  # Docker 无显示建议用 offscreen
    out_path=os.path.join(OUT_DIR, "viz_occ_plane_points.png"),
    plane_z=0.0,                       # 平面系 z=0 处铺设
    point_downsample=800_000
)

# # 4) 选择目标等距 d* 并从距离场抽取闭环
# d_min = float(cfg["free_space"]["d_min"])
# d_max = float(cfg["free_space"]["d_max"])
# r = float(cfg["free_space"]["d_target_ratio"])
# d_star = d_min*(1.0-r) + d_max*r
# band = float(cfg["free_space"]["band_width"])

# # 极点 = 点云的平面投影质心（也可以用最大密度处）
# center_uv = Xp[:,0:2].mean(axis=0)

# loop_uv = extract_closed_loop_from_distance(
#     D, origin_uv, cfg["free_space"]["grid_res"],
#     center_uv=center_uv,
#     d_star=d_star, band_width=band,
#     angles=int(cfg["free_space"]["polar_angles"])
# )
# if loop_uv is None or loop_uv.shape[0] < 50:
#     raise RuntimeError("自由空间闭环提取失败：请调大 band / 调整 d_min/d_max / 放宽 grid_res")

# # 5) 圆滑+均匀重采样（闭环）
# loop_uv_smooth = resample_smooth_closed_curve(
#     loop_uv,
#     samples=int(cfg["free_space"]["loop_samples"]),
#     smooth=float(cfg["free_space"]["spline_smooth"])
# )

# # 6) 抬回3D并给出初始朝向（混合：沿切线 + 看向质心）
# poses_all, loop_xyz = lift_loop_and_orient(
#     loop_uv_smooth,
#     plane_to_world=plane_to_world,
#     plane_height_offset=float(cfg["free_space"]["height_offset"]),
#     look_mode="mixed", alpha=0.35,
#     up=np.array([0,0,1.0], dtype=np.float32)
# )

# 4) 选择目标等距 d* 并从距离场抽取闭环
d_min = float(cfg["free_space"]["d_min"])
d_max = float(cfg["free_space"]["d_max"])
r = float(cfg["free_space"]["d_target_ratio"])
d_star = d_min*(1.0-r) + d_max*r
band = float(cfg["free_space"]["band_width"])

# 极点：bbox 内点的平面质心
center_uv = Xp[:,0:2].mean(axis=0)

loop_uv = extract_closed_loop_from_distance(
    D, origin_uv, cfg["free_space"]["grid_res"],
    center_uv=center_uv,
    d_star=d_star, band_width=band,
    angles=int(cfg["free_space"]["polar_angles"])
)
if loop_uv is None or loop_uv.shape[0] < 50:
    raise RuntimeError("自由空间闭环提取失败：请调大 band / 调整 d_min/d_max / 放宽 grid_res")

# 5) 圆滑+均匀重采样（闭环）
loop_uv_smooth = resample_smooth_closed_curve(
    loop_uv,
    samples=int(cfg["free_space"]["loop_samples"]),
    smooth=float(cfg["free_space"]["spline_smooth"])
)

# # 6) 抬回3D并给出初始朝向
# poses_all, loop_xyz = lift_loop_and_orient(
#     loop_uv_smooth,
#     plane_to_world=plane_to_world,
#     plane_height_offset=float(cfg["free_space"]["height_offset"]),
#     look_mode="mixed", alpha=0.35,
#     up=np.array([0,0,1.0], dtype=np.float32)
# )

# # 7) 基于重叠 & 可见点阈值进行二次采样，得到最终 poses
# poses, img_paths = [], []
# MIN_VIS = int(cfg["sampling"]["min_visible"])
# K_torch = K_torch  # 你已有 (1,3,3)
# W, H = int(W), int(H)

# K_np = K_torch[0].detach().cpu().numpy()

# visualize_path_open3d(scene, poses, K_np, W, H,
#                       loop_xyz=loop_xyz,
#                       cam_every=max(1, len(poses)//30),
#                       cam_size=0.6,
#                       mode="window",      # 或 "window"
#                       out_path=None)

# # 从第一个达标位姿起步
# start_idx = None
# for i in range(0, min(50, len(poses_all))):
#     R, t = poses_all[i]
#     T = np.eye(4, dtype=np.float32); T[:3,:3]=R; T[:3,3]=t
#     T_1x44 = torch.tensor(T, dtype=torch.float32, device=DEVICE).unsqueeze(0)
#     vis, z_med = visible_count(means, K_torch, T_1x44, W, H,
#                                sample_N=cfg["sampling"]["vis_sample_max"])
#     if vis >= MIN_VIS and z_med is not None and (z_med >= d_min) and (z_med <= d_max*1.5):
#         poses.append((R,t)); img_paths.append(f"images/loop_{len(img_paths):04d}.png")
#         cur_zmed = z_med; start_idx = i
#         break
# if start_idx is None:
#     raise RuntimeError("闭环起点附近可见点过少或距离不合适，请调整 d_min/d_max/band/grid_res")

# # 沿闭环按 overlap 前进（动态步长）
# i = start_idx
# visited = 0
# while visited < len(poses_all)-1:
#     # 根据上一帧深度中位数算步长
#     step = step_from_overlap(cfg["render"]["hfov_deg"], cfg["render"]["overlap"],
#                              z_med=cur_zmed, safety=cfg["sampling"]["step_safety"])
#     # 在 loop_xyz 上按弧长前进 step
#     acc = 0.0
#     j = (i+1) % len(loop_xyz)
#     while True:
#         seg = np.linalg.norm(loop_xyz[j] - loop_xyz[(j-1)%len(loop_xyz)])
#         if acc + seg >= step: break
#         acc += seg
#         j = (j+1) % len(loop_xyz)
#         if j == i: break  # 兜一圈了
#     # 插值得到候选相机位置 & 姿态（保持 poses_all 的朝向）
#     prev = loop_xyz[(j-1)%len(loop_xyz)]
#     nextp= loop_xyz[j]
#     ratio = (step - acc) / max(1e-6, np.linalg.norm(nextp - prev))
#     cand = prev + ratio * (nextp - prev)
#     # 用最近的原始朝向（避免重新计算）
#     R, t = poses_all[j]
#     # 以 cand 替换该姿态的位置（只平移不旋），保证圆滑
#     # 重新计算 t = -R*cam
#     t = -R @ cand.astype(np.float32)

#     T = np.eye(4, dtype=np.float32); T[:3,:3]=R; T[:3,3]=t
#     T_1x44 = torch.tensor(T, dtype=torch.float32, device=DEVICE).unsqueeze(0)
#     vis, z_med = visible_count(means, K_torch, T_1x44, W, H,
#                                sample_N=cfg["sampling"]["vis_sample_max"])
#     if vis >= MIN_VIS and (z_med is not None) and (d_min <= z_med <= d_max):
#         poses.append((R,t))
#         img_paths.append(f"images/loop_{len(img_paths):04d}.png")
#         cur_zmed = z_med
#         i = j
#     else:
#         # 如果不达标，微前进一小段
#         i = j
#     visited += 1

# ---------- 6) 只返回3D坐标 ----------
loop_xyz = lift_loop_positions(
    loop_uv_smooth, plane_to_world=plane_to_world,
    plane_height_offset=float(cfg["free_space"]["height_offset"])
)

# ---------- 7) 按重叠率在路径上等距采样（不涉及朝向） ----------
d_min   = float(cfg["free_space"]["d_min"])
d_max   = float(cfg["free_space"]["d_max"])
r_ratio = float(cfg["free_space"]["d_target_ratio"])
d_target = d_min*(1.0-r_ratio) + d_max*r_ratio

step_len = step_from_overlap_static(
    hfov_deg=cfg["render"]["hfov_deg"],
    overlap=cfg["render"]["overlap"],
    z_target=d_target
)
cam_positions, idx_hints = sample_positions_along_loop(loop_xyz, step_len)

# ---------- 8) 在每个采样点上搜索朝向；若不达标则微移 ----------
MIN_VIS   = int(cfg["sampling"]["min_visible"])
W, H      = int(W), int(H)
world_pts_np = scene_box['xyz'].detach().cpu().numpy()
poses = []
img_paths = []

shift_step  = float(cfg["sampling"].get("refine_shift_step", 0.3*step_len))
refine_tries= int(cfg["sampling"].get("refine_tries", 3))

for n, (cam, ih) in enumerate(zip(cam_positions, idx_hints)):
    i0 = int(ih) % len(loop_xyz)
    prev_pt = loop_xyz[(i0-1+len(loop_xyz))%len(loop_xyz)]
    next_pt = loop_xyz[(i0+1)%len(loop_xyz)]

    R, t, vis, z, ok = choose_orientation_for_cam(
        cam, prev_pt, next_pt,
        world_pts_np,
        means, K_torch, W, H, DEVICE,
        d_min, d_max, d_target,
        MIN_VIS, cfg["sampling"]["vis_sample_max"]
    )
    if not ok:
        R, t, cam_new, vis, z, ok = refine_cam_by_shifting(
            loop_xyz, ih, cam, shift_step, refine_tries,
            world_pts_np,
            means, K_torch, W, H, DEVICE,
            d_min, d_max, d_target, MIN_VIS, cfg["sampling"]["vis_sample_max"]
        )
        cam = cam_new  # 更新位置（可能微动）

    poses.append((R, t))
    img_paths.append(f"images/loop_{n:04d}.png")

print(f"[traj] 采样完成，共 {len(poses)} 个姿态")
K_np = K_torch[0].detach().cpu().numpy()   # (3,3)

visualize_path_open3d(scene, poses, K_np, W, H,
                      loop_xyz=loop_xyz,
                      cam_every=max(1, len(poses)//30),
                      cam_size=0.6,
                      mode="window",      # 或 "window"
                      out_path=None)

# ---------------- 渲染并保存 ----------------
for (R, t), path in zip(poses, img_paths):

    print("Rendering", path)
    # 将R,t转为相机位姿（world->cam）。gsplat通常需要cam位姿或其逆：
    # 构造 4x4 世界->相机矩阵（R,t 已经是 world->cam）
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3,  3] = t
    viewmats = torch.tensor(T, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,4,4)

    imgs, meta_img, meta = gs.rasterization(
        means=means,           # (N,3) 
        quats=q_wxyz,          # (N,4) 
        scales=scales,         # (N,3) 
        opacities=opacs,       # (N,)   
        colors=colors,         # (N,3)  
        viewmats=viewmats,     # (1,4,4) 世界->相机
        Ks=K_torch,            # (1,3,3)
        width=W, height=H,
        render_mode="RGB",
        rasterize_mode="classic",   # 需要更平滑可用 "antialiased"
        radius_clip=0.0
    )


    rgb = (imgs[0].clamp(0,1).detach().cpu().numpy()*255).astype(np.uint8)  # (H,W,3)
    imageio.imwrite(os.path.join(OUT_DIR, path), rgb)

# ---------------- 写COLMAP文本 ----------------
with open(os.path.join(OUT_DIR, "cameras.txt"), "w") as f:
    # SIMPLE_PINHOLE: model id=4; params: fx, cx, cy
    # 或 PINHOLE(id=1; fx, fy, cx, cy)。这里用 PINHOLE:
    f.write("# Camera list with one line of data per camera:\n")
    f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    f.write("1 PINHOLE {} {} {} {} {} {}\n".format(W, H, fx, fy, cx, cy))

with open(os.path.join(OUT_DIR, "images.txt"), "w") as f:
    f.write("# Image list with two lines of data per image:\n")
    f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
    for idx, ((R, t), path) in enumerate(zip(poses, img_paths), start=1):
        qw, qx, qy, qz, tx, ty, tz = colmap_qt_from_Rt(R, t)
        f.write(f"{idx} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {path}\n\n")

# 调用前转成 (3,3) numpy
K_np = K_torch[0].detach().cpu().numpy()   # (3,3)

# points3D.txt 可留空文件占位
open(os.path.join(OUT_DIR, "points3D.txt"), "w").close()
print("Done.")
# viz_open3d(scene, poses, Kpython3 manual_plane.py --gaussians /workspace/src/data/point_cloud_gs.ply --fov 70 --imgw 1920 --imgh 1080 --near 0.1 --far 20.0 --min_visible 800 --overlap_ratio 0.4 --ds 0.3_np, W, H, bbox_min=bbox_min, bbox_max=bbox_max,
#            max_points=800000, cam_every=max(1,len(poses)//30 or 1), cam_size=0.5)
