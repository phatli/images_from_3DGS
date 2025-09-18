import numpy as np
import torch
import time

 # ===== 基于体素密度找高密度 ROI =====
def density_roi(xyz_np, voxel=0.2, min_count=30, keep_top_percent=0.8):
    """
    xyz_np: (N,3) numpy
    voxel: 体素边长（按你场景尺度调，室内可 0.05~0.2，室外 0.2~1.0）
    min_count: 体素内最少点数阈值（太稀疏就不要）
    keep_top_percent: 只保留高密度体素的前 P 分位
    返回：roi_min, roi_max（numpy 3,）
    """
    print(*"正在计算 ROI（基于体素密度）...".split())
    
    mins = xyz_np.min(axis=0) - 1e-6
    maxs = xyz_np.max(axis=0) + 1e-6
    bins = np.ceil((maxs - mins) / voxel).astype(int)
    # 防止 bins 过大
    bins = np.clip(bins, 1, 2048)
    H, edges = np.histogramdd(xyz_np, bins=bins, range=[[mins[0], maxs[0]], [mins[1], maxs[1]], [mins[2], maxs[2]]])
    counts = H.flatten()
    if counts.size == 0 or counts.max() == 0:
        return mins, maxs  # 兜底
    # 只保留高密体素
    thresh = max(min_count, np.quantile(counts[counts>0], keep_top_percent))
    mask = (H >= thresh)
    if not np.any(mask):
        # 放宽标准
        mask = (H >= max(1, int(min_count/2)))
    idxs = np.argwhere(mask)
    # 体素 -> 空间坐标
    x_edges, y_edges, z_edges = edges
    # 选中体素的边界集合
    xs_min = x_edges[idxs[:,0]]
    ys_min = y_edges[idxs[:,1]]
    zs_min = z_edges[idxs[:,2]]
    xs_max = x_edges[idxs[:,0]+1]
    ys_max = y_edges[idxs[:,1]+1]
    zs_max = z_edges[idxs[:,2]+1]
    roi_min = np.array([xs_min.min(), ys_min.min(), zs_min.min()], dtype=np.float32)
    roi_max = np.array([xs_max.max(), ys_max.max(), zs_max.max()], dtype=np.float32)

    print("ROI已经获得，角点空间坐标为:", roi_min, roi_max)
    return roi_min, roi_max

# ---------- ROI 内的骨架体素、连通与折线路径 ----------

def make_skeleton_polyline(xyz_np, roi_min, roi_max,
                           voxel_track=0.25,
                           min_component_ratio=0.02,
                           smooth_window=5,
                           progress=None,           # None / "tqdm" / "live"
                           live_every=200,          # live 模式下的刷新间隔
                           live_max_pts=200000      # live 下点云最大绘制点数
                           ):
    """
    返回 polyline: (M,3)
    """
    use_tqdm = (progress == "tqdm")
    use_live = (progress == "live")

    # ---------- 可选：初始化可视化 ----------
    if use_live:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        plt.ion()
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20., azim=60.)
        # 采样显示点云（ROI 内），减少绘制负担
        m = (xyz_np[:,0]>=roi_min[0])&(xyz_np[:,0]<=roi_max[0])&\
            (xyz_np[:,1]>=roi_min[1])&(xyz_np[:,1]<=roi_max[1])&\
            (xyz_np[:,2]>=roi_min[2])&(xyz_np[:,2]<=roi_max[2])
        Xshow = xyz_np[m]
        if Xshow.shape[0] > live_max_pts:
            idx = np.random.default_rng(0).choice(Xshow.shape[0], live_max_pts, replace=False)
            Xshow = Xshow[idx]
        scat = ax.scatter(Xshow[:,0], Xshow[:,1], Xshow[:,2], s=0.2, c='gray', alpha=0.6)
        line_plot = None
        ax.set_title("Skeleton building... (voxelization)")
        fig.canvas.draw(); fig.canvas.flush_events()

        def live_update_line(pts, title=None):
            nonlocal line_plot
            if line_plot is not None: 
                line_plot.remove()
            if pts is not None and len(pts) >= 2:
                line_plot, = ax.plot(pts[:,0], pts[:,1], pts[:,2], 'r-', linewidth=1.2, alpha=0.9)
            if title:
                ax.set_title(title)
            fig.canvas.draw(); fig.canvas.flush_events()

    # ---------- 阶段 1：体素化 ----------
    t0 = time.time()
    m = (xyz_np[:,0]>=roi_min[0])&(xyz_np[:,0]<=roi_max[0])&\
        (xyz_np[:,1]>=roi_min[1])&(xyz_np[:,1]<=roi_max[1])&\
        (xyz_np[:,2]>=roi_min[2])&(xyz_np[:,2]<=roi_max[2])
    X = xyz_np[m]
    if X.shape[0] < 1000:
        X = xyz_np  # 兜底：点太少时退回全局

    mins = X.min(axis=0) - 1e-6
    idx = np.floor((X - mins) / voxel_track).astype(np.int64)  # (M,3)

    # unique + inverse 映射
    if use_tqdm:
        from tqdm import tqdm
        pbar = tqdm(total=3, desc="Voxelization", leave=False)
    uniq, inv = np.unique(idx, axis=0, return_inverse=True)   # 体素索引 (V,3)
    counts = np.bincount(inv)                                  # 每个体素点数
    centers = mins + (uniq + 0.5) * voxel_track                # 体素中心 (V,3)
    if use_tqdm:
        pbar.update(1)

    # ---------- 阶段 2：连通分量（6邻接 BFS） ----------
    voxset = set(map(tuple, uniq.tolist()))
    # 邻接缓存：提升 BFS 速度
    nbr_cache = {}
    def neighbors(v):
        if v in nbr_cache: return nbr_cache[v]
        i,j,k = v
        cand = [(i+1,j,k),(i-1,j,k),(i,j+1,k),(i,j-1,k),(i,j,k+1),(i,j,k-1)]
        out = [w for w in cand if w in voxset]
        nbr_cache[v] = out
        return out

    uniq_tuples = list(map(tuple, uniq.tolist()))
    visited = set()
    comps = []
    V = len(uniq_tuples)

    if use_tqdm:
        pbar2 = tqdm(total=V, desc="Connected Components", leave=False)

    for v in uniq_tuples:
        if v in visited:
            if use_tqdm: pbar2.update(1)
            continue
        q=[v]; visited.add(v); comp=[v]
        while q:
            cur = q.pop()
            for nb in neighbors(cur):
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb); comp.append(nb)
        comps.append(comp)
        if use_tqdm:
            # 这一步无法精确计数已处理节点数，近似按当前 visited 数增长推进
            pbar2.n = len(visited); pbar2.refresh()

        if use_live and (len(visited) % live_every == 0):
            # live 模式下绘制“已发现分量的质心连线”（粗略显示进度）
            centers_found = []
            for c in comps:
                arr = np.array(c)
                pts = mins + (arr + 0.5) * voxel_track
                centers_found.append(pts.mean(axis=0))
            if len(centers_found) >= 2:
                approx_poly = np.array(centers_found)
                live_update_line(approx_poly, title=f"Connected Components: {len(visited)}/{V}")

    if use_tqdm:
        pbar2.close()
        pbar.update(1)

    # 过滤小分量
    sizes = np.array([len(c) for c in comps], dtype=int)
    keep = sizes >= max(1, int(sizes.sum()*min_component_ratio))
    kept = [c for c,k in zip(comps, keep) if k]
    if len(kept)==0: kept = [max(comps, key=len)]

    # ---------- 阶段 3：串联每个大分量为折线（贪心最近邻） ----------
    def comp_center(comp):
        arr = np.array(comp)
        pts = mins + (arr + 0.5) * voxel_track
        return pts.mean(axis=0)

    kept.sort(key=lambda c: np.linalg.norm(comp_center(c) - centers.mean(axis=0)))
    poly = []

    if use_tqdm:
        total_pts = sum(len(c) for c in kept)
        pbar3 = tqdm(total=total_pts, desc="Greedy chaining", leave=False)
        done_pts = 0

    for ci, comp in enumerate(kept):
        arr = np.array(comp)
        pts = mins + (arr + 0.5) * voxel_track    # (K,3)
        # 简单的 O(K^2) 贪心最近邻串联（K 很大会慢，但直观）
        used = np.zeros(len(pts), dtype=bool)
        cur = 0
        order = [cur]; used[cur]=True

        # 逐步可视化
        for step in range(1, len(pts)):
            # 只在未使用点中找最近
            not_used = np.where(~used)[0]
            d = np.linalg.norm(pts[not_used] - pts[cur], axis=1)
            j_local = np.argmin(d)
            j = not_used[j_local]
            order.append(j); used[j]=True; cur=j

            if use_tqdm:
                done_pts += 1
                if step % 50 == 0:
                    pbar3.n = done_pts; pbar3.refresh()

            if use_live and (step % live_every == 0):
                poly_tmp = np.concatenate([np.array(p) for p in poly], axis=0) if len(poly)>0 else None
                this_seg = pts[order]
                if poly_tmp is None:
                    live_update_line(this_seg, title=f"Chaining comp {ci+1}/{len(kept)} - step {step}/{len(pts)}")
                else:
                    live_update_line(np.vstack([poly_tmp, this_seg]),
                                     title=f"Chaining comp {ci+1}/{len(kept)} - step {step}/{len(pts)}")

        poly.append(pts[order])

    if use_tqdm:
        pbar3.n = total_pts; pbar3.refresh(); pbar3.close()
        pbar.update(1); pbar.close()

    polyline = np.concatenate(poly, axis=0)  # (M,3)

    # ---------- 阶段 4：平滑 ----------
    if smooth_window > 1 and polyline.shape[0] >= smooth_window:
        k = smooth_window
        ker = np.ones(k)/k
        # 显示平滑进度
        if use_tqdm:
            from tqdm import tqdm
            pbar4 = tqdm(total=3, desc="Smoothing", leave=False)
        for a in range(3):
            polyline[:,a] = np.convolve(polyline[:,a], ker, mode='same')
            if use_tqdm: pbar4.update(1)
            if use_live:
                live_update_line(polyline, title=f"Smoothing {a+1}/3")
        if use_tqdm: pbar4.close()

    if use_live:
        live_update_line(polyline, title="Done")
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()

    return polyline

# ---------- 可见点数量估计（CUDA） ----------
@torch.no_grad()
def visible_count(means_t, K_1x33, T_1x44, W, H, sample_N=300000):
    N = means_t.shape[0]
    if N > sample_N:
        idx = torch.randint(0, N, (sample_N,), device=means_t.device)
        P = means_t[idx]
    else:
        P = means_t
    ones = torch.ones((P.shape[0],1), device=P.device, dtype=P.dtype)
    X = torch.cat([P, ones], dim=1).T  # (4,M)
    Pmat = torch.bmm(K_1x33, T_1x44[:,:3,:])  # (1,3,4)
    x = (Pmat @ X).squeeze(0)                 # (3,M)
    zc = x[2]
    u = x[0] / (zc + 1e-8)
    v = x[1] / (zc + 1e-8)
    mask = (zc > 0) & (u>=0) & (u<W) & (v>=0) & (v<H)
    return int(mask.sum().item()), zc[mask].median().item() if mask.any() else None

# ---------- 根据上一帧的“可见深度中位数”计算步长，满足目标 overlap ----------
def step_from_overlap(hfov_deg, overlap, z_med, safety=0.8):
    HFOV = np.deg2rad(hfov_deg)
    s = 2.0 * z_med * np.tan(HFOV/2.0) * (1.0 - overlap)
    return max(1e-4, float(s * safety))  # 避免0步长

# ---------- 生成朝向（look-at：沿切线方向看/或看向局部质心） ----------
def pose_from_tangent(cam, tangent, up=np.array([0,0,1.0], dtype=np.float32)):
    t = tangent / (np.linalg.norm(tangent)+1e-9)
    z = -t  # 相机前向（世界->相机里，z 轴指向视线方向）
    x = np.cross(up, z); x = x/ (np.linalg.norm(x)+1e-9)
    y = np.cross(z, x)
    R = np.stack([x,y,z], axis=0)  # world->cam
    tvec = -R @ cam
    return R, tvec
