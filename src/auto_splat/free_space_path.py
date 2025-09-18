import numpy as np
import scipy.ndimage as ndi

# ========== 主平面拟合（PCA）并建立局部坐标 ==========
def fit_principal_plane(xyz):
    # 去均值 + SVD (3x3) → 最小奇异向量是法线
    mu = xyz.mean(axis=0)
    X = xyz - mu
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    n = Vt[-1]                         # 法线（世界系）
    n = n / (np.linalg.norm(n) + 1e-9)
    # 构造平面基 u,v 使 [u,v,n] 正交
    up = np.array([0,0,1.0], dtype=np.float32)
    if abs(np.dot(up, n)) > 0.95:
        up = np.array([0,1,0], dtype=np.float32)
    u = np.cross(up, n); u /= (np.linalg.norm(u)+1e-9)
    v = np.cross(n, u)
    Rpw = np.stack([u, v, n], axis=0)  # world->plane 的旋转（行是基）
    # 投影函数
    def world_to_plane(P):
        return (Rpw @ (P - mu).T).T    # (N,3) -> (N,3)，其中 z 是法向距离
    def plane_to_world(Pp):
        return (Rpw.T @ Pp.T).T + mu
    return mu, Rpw, world_to_plane, plane_to_world, n

# ========== 估计“占据半径” ==========
def gaussian_radius_from_scale(scales_np, gain=2.5):
    """
    scales_np: (N,3) 是 exp(scale) 后的线性尺寸(σx,σy,σz)。
    返回 2D 投影时的等效半径（取均值σ并乘系数），粗略近似足够用于自由空间。
    """
    sig = np.mean(scales_np, axis=1)   # (N,)
    r = gain * sig
    return r

# ========== 建 2D 栅格占据图 & 距离场 ==========
def build_2d_occupancy_and_distance(uvr, grid_res, margin=3):
    """
    uvr: (N,3) -> (u,v,r)  其中 r 是占据半径（米）
    grid_res: 每格米数
    margin: 额外边界格数
    返回：occ(bool HxW), D(float HxW), origin(np.array([u0,v0])), (H,W)
    """
    U = uvr[:,0]; V = uvr[:,1]; R = uvr[:,2]
    umin, vmin = np.min(U-R), np.min(V-R)
    umax, vmax = np.max(U+R), np.max(V+R)
    W = int(np.ceil((umax - umin) / grid_res)) + 1 + 2*margin
    H = int(np.ceil((vmax - vmin) / grid_res)) + 1 + 2*margin
    u0 = umin - margin*grid_res
    v0 = vmin - margin*grid_res
    occ = np.zeros((H, W), dtype=bool)

    # 把每个圆盘 rasterize 到栅格（朴素填充，快且够用）
    for u, v, r in zip(U, V, R):
        if r <= 0: continue
        ru = int(np.ceil(r / grid_res))
        cx = int((u - u0) / grid_res)
        cy = int((v - v0) / grid_res)
        y0, y1 = max(0, cy-ru), min(H-1, cy+ru)
        x0, x1 = max(0, cx-ru), min(W-1, cx+ru)
        ys = np.arange(y0, y1+1)
        xs = np.arange(x0, x1+1)
        YY, XX = np.meshgrid(ys, xs, indexing='ij')
        du = (XX*grid_res + u0) - u
        dv = (YY*grid_res + v0) - v
        occ[YY, XX] |= (du*du + dv*dv) <= (r*r)

    # 自由空间距离场：对 ~occ 做 EDT，单位转成“米”
    D_pix = ndi.distance_transform_edt(~occ)    # 像素单位
    D = D_pix * grid_res
    return occ, D, np.array([u0, v0], dtype=np.float32), (H, W)

# ========== 在距离场上抽取闭环 ==========
def extract_closed_loop_from_distance(D, origin_uv, grid_res,
                                      center_uv, d_star, band_width=0.5,
                                      angles=2048):
    """
    沿极坐标射线在距离场上寻找距离 ≈ d_star 的交点，形成一条闭合曲线（uv）。
    band_width: 允许的带宽（±band/2），保证连贯性。
    center_uv: 以哪个平面点为极点（可用点云投影的质心）
    """
    H, W = D.shape
    u0, v0 = origin_uv
    thetas = np.linspace(0, 2*np.pi, angles, endpoint=False)
    uv_list = []

    for th in thetas:
        # 在一根射线上用“从近到远”的方式搜索 |D - d_star| 最小且落在band内的点
        # 把射线离散为固定步长（一个像素）
        # 初始点：center_uv
        u_c, v_c = center_uv
        # 从 d = max(0, d_star - band/2) 到 d = d_star + band/2
        d_min = max(0.0, d_star - band_width*0.5)
        d_max = d_star + band_width*0.5
        found = None
        # 按“像素步长”前进
        step_pix = 1.0
        # 把米换像素步进量
        step_u = np.cos(th) * grid_res
        step_v = np.sin(th) * grid_res
        # 从 center 开始向外试探，最多走 max(H,W) 步
        max_steps = int(max(H, W) * 1.2)
        u, v = u_c, v_c
        for _ in range(max_steps):
            # 栅格坐标
            xi = int((u - u0) / grid_res)
            yi = int((v - v0) / grid_res)
            if xi < 0 or xi >= W or yi < 0 or yi >= H:
                break
            d_here = D[yi, xi]
            if d_here >= d_min and d_here <= d_max:
                # 记录“最靠近 d_star”的位置
                if (found is None) or (abs(d_here - d_star) < found[2]):
                    found = (u, v, abs(d_here - d_star))
                # 继续走，看有没有更近的
            # 走一步
            u += step_u
            v += step_v
        if found is not None:
            uv_list.append([found[0], found[1]])

    uv = np.array(uv_list, dtype=np.float32)
    # 如果有缺口（某些角度找不到），用邻近角度线性插值修补
    # 简化处理：若比例过低，返回空
    if uv.shape[0] < angles * 0.6:
        return None
    # 尽量按角度顺序保留（已按 th 顺序）
    return uv

# ========== B样条圆滑 & 均匀重采样 ==========
def resample_smooth_closed_curve(uv, samples=400, smooth=0.0):
    """
    uv: (M,2) 闭合曲线（按序）
    返回：等弧长重采样后的闭合曲线 (samples, 2)
    """
    from scipy.interpolate import splprep, splev
    # 保证闭合：首尾拼接
    P = np.vstack([uv, uv[0:1]])
    # 计算累积弧长
    ds = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(ds)])
    # 归一化到 [0,1]
    t = s / s[-1] if s[-1] > 0 else np.linspace(0, 1, len(P))

    # B样条拟合（周期）
    per = True
    k = 3 if len(P) > 4 else min(3, len(P)-1)
    tck, _ = splprep([P[:,0], P[:,1]], u=t, s=smooth*len(P), per=per, k=k)
    tt = np.linspace(0, 1, samples+1)[:-1]  # 丢弃最后一个点保持闭环不重复
    uu, vv = splev(tt, tck)
    return np.stack([uu, vv], axis=1).astype(np.float32)

# ========== 2D 曲线抬回 3D 并设置相机朝向 ==========
def lift_loop_and_orient(loop_uv, plane_to_world, plane_height_offset,
                         look_mode="mixed", alpha=0.35, up=np.array([0,0,1],dtype=np.float32)):
    """
    loop_uv: (M,2)  → 变成 loop_xyz (M,3)
    look_mode: "tangent" / "centroid" / "mixed"
    alpha: 混合权重（mixed时）
    返回：poses=[(R,t)], loop_xyz
    """
    M = loop_uv.shape[0]
    # 抬回 3D：给每个点加一个恒定高度偏置
    loop_p = np.zeros((M,3), dtype=np.float32)
    loop_p[:,0:2] = loop_uv
    loop_p[:,2] = plane_height_offset
    loop_xyz = plane_to_world(loop_p)

    # 用全局质心作为 look-at 参考
    centroid = loop_xyz.mean(axis=0)

    poses = []
    for i in range(M):
        cam = loop_xyz[i]
        nxt = loop_xyz[(i+1)%M]
        prv = loop_xyz[(i-1+M)%M]
        tangent = nxt - prv
        tdir = tangent / (np.linalg.norm(tangent)+1e-9)

        if look_mode == "tangent":
            z = tdir
        elif look_mode == "centroid":
            z = (centroid - cam); z /= (np.linalg.norm(z)+1e-9)
        else:
            z_t = tdir
            z_c = (centroid - cam); z_c /= (np.linalg.norm(z_c)+1e-9)
            z = (1.0-alpha)*z_t + alpha*z_c
            z /= (np.linalg.norm(z)+1e-9)

        # 正交基
        x = np.cross(up, z); x /= (np.linalg.norm(x)+1e-9)
        y = np.cross(z, x)
        R = np.stack([x,y,z], axis=0).astype(np.float32)  # world->cam
        t = -R @ cam.astype(np.float32)
        poses.append((R,t))
    return poses, loop_xyz