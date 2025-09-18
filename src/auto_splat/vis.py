# ====== Open3D 可视化：点云 + 轨迹 + 相机视锥 ======
import numpy as np
import torch
import open3d as o3d
import os

def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

def build_pointcloud_o3d(scene, max_points=1_000_000, seed=0):
    xyz = to_np(scene['xyz'])
    rgb = to_np(scene['rgb']) if 'rgb' in scene else None
    N = xyz.shape[0]
    if N > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=max_points, replace=False)
        xyz = xyz[idx]; rgb = (rgb[idx] if rgb is not None else None)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb,0,1))
    return pcd

def frustum_mesh(K, W, H, size=0.5, color=[1,0,0]):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    pix = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)
    rays = np.stack([(pix[:,0]-cx)/fx, (pix[:,1]-cy)/fy, np.ones(4)], axis=1)
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    O = np.zeros((1,3))
    corners = O + rays * size
    # 线框
    pts = np.vstack([O, corners])  # 5 x 3
    lines = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
    colors = [color for _ in lines]
    ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pts),
                              lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def transform_lineset(ls, Rwc, twc):
    pts = np.asarray(ls.points)
    pts_w = (pts @ Rwc.T) + twc[None,:]
    ls.points = o3d.utility.Vector3dVector(pts_w)
    return ls

def build_trajectory_lines(traj_xyz, color=[0,0,1]):
    lines = [(i,i+1) for i in range(len(traj_xyz)-1)]
    ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(traj_xyz),
                              lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return ls

def build_bbox_lines(bmin, bmax, color=[0,0,0]):
    x0,y0,z0 = bmin; x1,y1,z1 = bmax
    P = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                  [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(P),
                              lines=o3d.utility.Vector2iVector(E))
    ls.colors = o3d.utility.Vector3dVector([color for _ in E])
    return ls

def viz_open3d(scene, poses, K, W, H, bbox_min, bbox_max,
               max_points, cam_every, cam_size):
    geoms = []
    pcd = build_pointcloud_o3d(scene, max_points=max_points)
    geoms.append(pcd)

    # 轨迹
    traj = []
    for (R,t) in poses:
        Rwc = R.T; twc = -Rwc @ t
        traj.append(twc)
    traj = np.array(traj)
    if len(traj)>=2:
        geoms.append(build_trajectory_lines(traj, color=[0,0,1]))

    # 相机视锥（抽稀画）
    for (R,t) in poses[::cam_every]:
        Rwc = R.T; twc = -Rwc @ t
        fr = frustum_mesh(K, W, H, size=cam_size, color=[1,0,0])
        geoms.append(transform_lineset(fr, Rwc, twc))

    # bbox
    if bbox_min is None or bbox_max is None:
        xyz = to_np(scene['xyz']); bbox_min, bbox_max = xyz.min(0), xyz.max(0)
    geoms.append(build_bbox_lines(bbox_min, bbox_max, color=[0,0,0]))

    o3d.visualization.draw_geometries(geoms)

def visualize_path_open3d(
    scene, poses, K, W, H,
    loop_xyz=None,
    max_points=800_000,
    cam_every=30,
    cam_size=0.6,
    mode="auto",          # "auto" | "window" | "offscreen"
    out_path=None         # 离屏时保存到此PNG路径
):
    """
    scene: dict，至少包含 'xyz' (Nx3)，可选 'rgb'
    poses: List[(R(3,3), t(3,))]，世界->相机
    K: (3,3) 或 (1,3,3)，torch/numpy 都可
    W,H: 图像宽高
    loop_xyz: (M,3) 闭环路径（可选）
    """
    import numpy as np
    import open3d as o3d
    from open3d.visualization import rendering

    # ---- K 统一 ----
    if "torch" in str(type(K)):
        K = K.detach().cpu().numpy()
    K = np.asarray(K)
    if K.ndim == 3:
        K = K[0]
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])

    # ---- 点云 ----
    xyz = scene['xyz'].detach().cpu().numpy() if hasattr(scene['xyz'], "device") else np.asarray(scene['xyz'])
    rgb = None
    if 'rgb' in scene and scene['rgb'] is not None:
        rgb = scene['rgb'].detach().cpu().numpy() if hasattr(scene['rgb'], "device") else np.asarray(scene['rgb'])
        rgb = np.clip(rgb, 0, 1)

    if xyz.shape[0] > max_points:
        sel = np.random.default_rng(0).choice(xyz.shape[0], max_points, replace=False)
        xyz = xyz[sel]
        if rgb is not None: rgb = rgb[sel]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    geoms = [("pcd", pcd)]

    # ---- 相机轨迹（蓝线） ----
    if len(poses) >= 2:
        traj_pts = []
        for (R, t) in poses:
            Rwc = R.T
            twc = -Rwc @ t
            traj_pts.append(twc)
        traj_pts = np.asarray(traj_pts, dtype=np.float32)
        lines = [(i, i+1) for i in range(len(traj_pts)-1)]
        traj_ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(traj_pts),
            lines=o3d.utility.Vector2iVector(lines)
        )
        traj_ls.colors = o3d.utility.Vector3dVector([[0,0,1] for _ in lines])
        geoms.append(("traj", traj_ls))

    # ---- 闭环（黑线） ----
    if loop_xyz is not None and loop_xyz.shape[0] >= 2:
        P = np.asarray(loop_xyz, dtype=np.float32)
        lines = [(i, (i+1) % len(P)) for i in range(len(P))]
        loop_ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(P),
            lines=o3d.utility.Vector2iVector(lines)
        )
        loop_ls.colors = o3d.utility.Vector3dVector([[0,0,0] for _ in lines])
        geoms.append(("loop", loop_ls))

    # ---- 相机视锥（红色线框） ----
    def frustum_lineset_single(K, W, H, scale=0.6, color=[1,0,0]):
        pix = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)
        rays = np.stack([(pix[:,0]-K[0,2])/K[0,0], (pix[:,1]-K[1,2])/K[1,1], np.ones(4)], axis=1)
        rays = rays / (np.linalg.norm(rays, axis=1, keepdims=True) + 1e-9)
        O = np.zeros((1,3), dtype=np.float32)
        corners = O + rays * scale
        segs = []
        for c in corners:
            segs.append((O[0], c))
        for a,b in [(0,1),(1,2),(2,3),(3,0)]:
            segs.append((corners[a], corners[b]))
        return np.asarray(segs, dtype=np.float32)

    cam_lines_all = []
    for idx in range(0, len(poses), max(1,int(cam_every))):
        R, t = poses[idx]
        segs_cam = frustum_lineset_single(K, W, H, scale=cam_size)
        Rwc = R.T; twc = -Rwc @ t
        segs_world = []
        for a,b in segs_cam:
            A = (Rwc @ a) + twc
            B = (Rwc @ b) + twc
            segs_world.append((A,B))
        cam_lines_all.extend(segs_world)

    if len(cam_lines_all) > 0:
        pts = np.vstack([np.asarray(segs_world).reshape(-1,3) for segs_world in [cam_lines_all]])
        lines = [(i, i+1) for i in range(0, pts.shape[0], 2)]
        cam_ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines)
        )
        cam_ls.colors = o3d.utility.Vector3dVector([[1,0,0] for _ in lines])
        geoms.append(("frusta", cam_ls))

    # ---- 显示/渲染 ----
    def show_window(geoms):
        o3d.visualization.draw_geometries([g for _,g in geoms])

    def render_offscreen(geoms, out_path):
        renderer = rendering.OffscreenRenderer(int(W), int(H))
        sc = renderer.scene
        sc.set_background([1,1,1,1])
        mat_pts = rendering.MaterialRecord(); mat_pts.shader = "defaultUnlit"; mat_pts.point_size = 1.0
        mat_line = rendering.MaterialRecord(); mat_line.shader = "unlitLine"; mat_line.line_width = 1.5
        for name, g in geoms:
            if isinstance(g, o3d.geometry.PointCloud):
                sc.add_geometry(name, g, mat_pts)
            else:
                sc.add_geometry(name, g, mat_line)
        img = renderer.render_to_image()
        if out_path is None: out_path="viz_o3d.png"
        o3d.io.write_image(out_path, img)
        print(f"[o3d-offscreen] saved {out_path}")

    env_headless = os.environ.get("OPEN3D_RENDERING_ENABLE_HEADLESS","0") == "1"
    if mode=="auto":
        mode = "offscreen" if (env_headless or out_path) else "window"

    if mode=="window":
        show_window(geoms)
    else:
        os.environ["OPEN3D_RENDERING_ENABLE_HEADLESS"]="1"
        render_offscreen(geoms, out_path)

def visualize_occ_plane_and_points_open3d(
    occ, D, origin_uv, grid_res,
    plane_to_world,              # 从 fit_principal_plane 得到
    scene,                       # dict，至少含 'xyz'，可含 'rgb'
    show_occupied=True,
    occ_alpha=0.35,
    downsample=1,                # 可视化降采样(>=1)
    mode="auto",                 # "auto" | "window" | "offscreen"
    out_path=None,               # 离屏保存路径
    plane_z=0.0,                 # 平面系 z 值（通常 0）
    point_downsample=800_000     # 点云最大点数
):
    """
    occ: (H,W) bool，占据为 True
    D:   (H,W) float，自由空间距离(米)
    origin_uv: [u0, v0]，对应 build_2d_occupancy_and_distance 的左上角
    grid_res: 每格米数
    plane_to_world: (Np,3) -> (Np,3)，把平面系点抬回世界系
    scene: {'xyz': (N,3)[torch/np], 'rgb':(N,3)[0~1] 可选}
    """
    import numpy as np
    import open3d as o3d
    from open3d.visualization import rendering

    H, W = occ.shape
    u0, v0 = float(origin_uv[0]), float(origin_uv[1])

    # ---- 颜色映射（小型 viridis 近似）----
    def viridis_like(x):
        x = np.clip(x, 0.0, 1.0)
        r = 0.2803 + 0.7200*x - 0.4200*x*x
        g = 0.1650 + 1.2000*x - 0.9000*x*x
        b = 0.4800 + 0.3000*x + 0.2000*x*x
        return np.clip(np.stack([r,g,b], axis=-1), 0.0, 1.0)

    geoms = []

    # ---- 自由空间点（~occ）按 D 上色 → 抬回世界系 ----
    mask_free = ~occ
    if downsample > 1:
        mask_free = mask_free[::downsample, ::downsample]
        D_show = D[::downsample, ::downsample]
    else:
        D_show = D

    ys, xs = np.where(mask_free)
    if ys.size > 0:
        U = xs * grid_res + u0 + 0.5*grid_res
        V = ys * grid_res + v0 + 0.5*grid_res
        Z = np.full_like(U, plane_z, dtype=np.float32)
        Pp = np.stack([U, V, Z], axis=1).astype(np.float32)     # 平面系
        Pw = plane_to_world(Pp)                                  # 世界系

        d_vals = D_show[mask_free]
        d_lo = np.quantile(d_vals, 0.02) if d_vals.size>0 else 0.0
        d_hi = np.quantile(d_vals, 0.98) if d_vals.size>0 else 1.0
        d_norm = (d_vals - d_lo) / max(1e-9, (d_hi - d_lo))
        C = viridis_like(d_norm)

        pcd_free = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pw))
        pcd_free.colors = o3d.utility.Vector3dVector(C)
        geoms.append(("free", pcd_free))

    # ---- 占据线框（True 栅格）→ 抬回世界系 ----
    if show_occupied:
        ys_o, xs_o = np.where(occ)
        if downsample > 1:
            ys_o = ys_o[::downsample]
            xs_o = xs_o[::downsample]

        lines_pts = []
        lines_idx = []
        base = 0
        for y, x in zip(ys_o, xs_o):
            cx = x * grid_res + u0 + 0.5*grid_res
            cy = y * grid_res + v0 + 0.5*grid_res
            s = 0.5*grid_res
            Pp = np.array([
                [cx-s, cy-s, plane_z],
                [cx+s, cy-s, plane_z],
                [cx+s, cy+s, plane_z],
                [cx-s, cy+s, plane_z]
            ], dtype=np.float32)
            Pw = plane_to_world(Pp)
            lines_pts.append(Pw)
            lines_idx += [(base+0, base+1), (base+1, base+2),
                          (base+2, base+3), (base+3, base+0)]
            base += 4
        if len(lines_idx) > 0:
            lines_pts = np.vstack(lines_pts)
            occ_ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(lines_pts),
                lines=o3d.utility.Vector2iVector(lines_idx)
            )
            occ_ls.colors = o3d.utility.Vector3dVector([[0.2+0.8*(1.0-occ_alpha)]*3 for _ in lines_idx])
            geoms.append(("occ", occ_ls))

    # ---- 原始点云（世界系）----
    xyz = scene['xyz']
    if hasattr(xyz, "device"):  # torch
        xyz = xyz.detach().cpu().numpy()
    else:
        xyz = np.asarray(xyz)
    rgb = None
    if 'rgb' in scene and scene['rgb'] is not None:
        rgb = scene['rgb']
        if hasattr(rgb, "device"): rgb = rgb.detach().cpu().numpy()
        rgb = np.clip(np.asarray(rgb), 0, 1)

    if xyz.shape[0] > point_downsample:
        sel = np.random.default_rng(0).choice(xyz.shape[0], point_downsample, replace=False)
        xyz = xyz[sel]; 
        if rgb is not None: rgb = rgb[sel]

    pcd_raw = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    if rgb is not None:
        pcd_raw.colors = o3d.utility.Vector3dVector(rgb)
    geoms.append(("pcd", pcd_raw))

    # ---- 渲染/显示 ----
    def show_window(geoms):
        o3d.visualization.draw_geometries([g for _,g in geoms])

    def render_offscreen(geoms, out_path):
        renderer = rendering.OffscreenRenderer(1280, 960)
        sc = renderer.scene
        sc.set_background([1,1,1,1])

        mat_pts = rendering.MaterialRecord(); mat_pts.shader = "defaultUnlit"; mat_pts.point_size = 1.1
        mat_line = rendering.MaterialRecord(); mat_line.shader = "unlitLine"; mat_line.line_width = 1.0

        for name, g in geoms:
            if isinstance(g, o3d.geometry.PointCloud):
                sc.add_geometry(name, g, mat_pts)
            else:
                sc.add_geometry(name, g, mat_line)

        # 视角：看向整体
        bounds = sc.bounding_box
        center = 0.5*(bounds.min_bound + bounds.max_bound)
        extent = np.linalg.norm(bounds.max_bound - bounds.min_bound)
        eye = center + np.array([0, -extent*0.8, extent*0.8])
        sc.camera.look_at(center, eye, [0,0,1])

        img = renderer.render_to_image()
        if out_path is None:
            out_path = "viz_occ_plane_points.png"
        o3d.io.write_image(out_path, img)
        print(f"[o3d-offscreen] saved {out_path}")

    env_headless = os.environ.get("OPEN3D_RENDERING_ENABLE_HEADLESS","0") == "1"
    if mode == "auto":
        mode = "offscreen" if (env_headless or out_path is not None) else "window"

    if mode == "window":
        show_window(geoms)
    else:
        os.environ["OPEN3D_RENDERING_ENABLE_HEADLESS"] = "1"
        render_offscreen(geoms, out_path)
