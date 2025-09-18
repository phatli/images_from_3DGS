# test_viz.py
# 用法示例：
# 1) 仅用 Matplotlib 离线保存：       python3 test_viz.py --backend mpl
# 2) Open3D 纯离屏保存PNG：           OPEN3D_RENDERING_ENABLE_HEADLESS=1 python3 test_viz.py --backend o3d-offscreen
# 3) Open3D 窗口交互（需要X11或xvfb）： python3 test_viz.py --backend o3d
#    若在无桌面服务器：xvfb-run -s "-screen 0 1920x1080x24" python3 test_viz.py --backend o3d

import os, argparse, numpy as np

def make_dummy_pointcloud(n=20000, seed=0):
    """生成一个小场景点云：一个平面 + 一条弧形墙 + 少量随机点"""
    rng = np.random.default_rng(seed)
    # 地面平面
    x = rng.uniform(-3, 3, n//2)
    y = rng.uniform(-3, 3, n//2)
    z = rng.normal(0.0, 0.01, n//2)
    P1 = np.stack([x, y, z], axis=1)
    # 弧形墙
    th = rng.uniform(-1.0, 1.0, n//3) * np.pi
    r = rng.uniform(1.5, 2.5, n//3)
    xx = r*np.cos(th)
    yy = r*np.sin(th)
    zz = rng.uniform(0.0, 2.0, n//3)
    P2 = np.stack([xx, yy, zz], axis=1)
    # 噪声点
    P3 = rng.uniform([-3,-3,0],[3,3,2.2], (n - (n//2 + n//3), 3))
    P = np.concatenate([P1,P2,P3], axis=0).astype(np.float32)
    # 颜色（根据高度着色）
    h = (P[:,2] - P[:,2].min()) / max(1e-6, (P[:,2].ptp()))
    rgb = np.stack([0.2+0.8*h, 0.4*(1-h)+0.2, 0.8*(1-h)], axis=1).astype(np.float32)
    return P, rgb

def build_dummy_poses():
    """构造三个位姿用于画视锥（世界->相机）"""
    def look_at(cam, target=np.array([0,0,0.5],dtype=np.float32), up=np.array([0,0,1],dtype=np.float32)):
        z = (target - cam); z /= (np.linalg.norm(z)+1e-9)
        x = np.cross(up, z); x /= (np.linalg.norm(x)+1e-9)
        y = np.cross(z, x)
        R = np.stack([x,y,z], axis=0).astype(np.float32)
        t = -R @ cam.astype(np.float32)
        return R, t
    cams = [np.array([4,  0, 2], np.float32),
            np.array([0,  4, 2], np.float32),
            np.array([-4, 0, 2], np.float32)]
    poses = [look_at(c) for c in cams]
    return poses

def pinhole_K(w=1280, h=960, hfov_deg=60.0):
    hfov = np.deg2rad(hfov_deg)
    fx = w/(2*np.tan(hfov/2)); fy = fx
    cx, cy = w/2, h/2
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
    return K

# ---------- Matplotlib 可视化（离线保存PNG） ----------
def viz_mpl(points, colors, poses, K, W, H, out_path):
    os.environ["MPLBACKEND"] = "Agg"  # 无GUI环境必须
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    P = points; C = colors

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:,0], P[:,1], P[:,2], s=0.2, c=np.clip(C,0,1), alpha=0.8)

    # 画视锥
    def frustum_lines(K, W, H, scale=0.6):
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        pix = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)
        rays = np.stack([(pix[:,0]-cx)/fx, (pix[:,1]-cy)/fy, np.ones(4)], axis=1)
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        O = np.zeros((1,3))
        corners = O + rays * scale
        segs = []
        for c in corners: segs.append((O[0], c))
        for a,b in [(0,1),(1,2),(2,3),(3,0)]: segs.append((corners[a], corners[b]))
        return np.array(segs)

    segs_cam = frustum_lines(K, W, H, scale=0.8)
    segs_world = []
    traj = []
    for (R,t) in poses:
        Rwc = R.T; twc = -Rwc @ t
        traj.append(twc)
        for a,b in segs_cam:
            A = (Rwc @ a) + twc
            B = (Rwc @ b) + twc
            segs_world.append((A,B))
    if len(traj)>=2:
        traj = np.array(traj)
        ax.plot(traj[:,0], traj[:,1], traj[:,2], 'b-', lw=1.5)

    lc = Line3DCollection(segs_world, colors='r', linewidths=0.8)
    ax.add_collection3d(lc)

    # 视域
    allp = np.vstack([P, traj]) if len(poses)>0 else P
    mn, mx = allp.min(0), allp.max(0); ctr = 0.5*(mn+mx); span = (mx-mn).max()*0.55
    ax.set_xlim(ctr[0]-span, ctr[0]+span)
    ax.set_ylim(ctr[1]-span, ctr[1]+span)
    ax.set_zlim(ctr[2]-span, ctr[2]+span)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=20., azim=60.)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[mpl] saved {out_path}")

# ---------- Open3D 离屏渲染（保存PNG，不创建窗口） ----------
def viz_o3d_offscreen(points, colors, poses, K, W, H, out_path):
    os.environ["OPEN3D_RENDERING_ENABLE_HEADLESS"] = "1"
    import open3d as o3d
    from open3d.visualization import rendering
    P = points; C = colors

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    if C is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(C,0,1))

    # 轨迹线
    traj = []
    for (R,t) in poses:
        Rwc = R.T; twc = -Rwc @ t
        traj.append(twc)
    traj_ls = None
    if len(traj)>=2:
        traj = np.array(traj)
        lines = [(i,i+1) for i in range(len(traj)-1)]
        traj_ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(traj),
                                       lines=o3d.utility.Vector2iVector(lines))
        traj_ls.colors = o3d.utility.Vector3dVector([[0,0,1] for _ in lines])

    renderer = rendering.OffscreenRenderer(W, H)
    sc = renderer.scene
    sc.set_background([1,1,1,1])

    mat_pts = rendering.MaterialRecord()
    mat_pts.shader = "defaultUnlit"
    mat_pts.point_size = 1.0
    sc.add_geometry("pcd", pcd, mat_pts)
    if traj_ls is not None:
        mat_line = rendering.MaterialRecord()
        mat_line.shader = "unlitLine"
        mat_line.line_width = 1.5
        sc.add_geometry("traj", traj_ls, mat_line)

    # 用第一帧设置相机
    if len(poses)==0:
        raise RuntimeError("poses 为空")
    R, t = poses[0]
    Rwc = R.T; twc = -Rwc @ t
    extr = np.eye(4, dtype=np.float32); extr[:3,:3]=Rwc; extr[:3,3]=twc

    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(W, H, K[0,0], K[1,1], K[0,2], K[1,2])
    sc.camera.setup(intr, extr)
    # 居中查看
    bounds = sc.bounding_box
    center = 0.5*(bounds.min_bound + bounds.max_bound)
    sc.camera.look_at(center, twc, [0,0,1])

    img = renderer.render_to_image()
    o3d.io.write_image(out_path, img)
    print(f"[o3d-offscreen] saved {out_path}")

# ---------- Open3D 窗口交互（需要X11/Wayland或xvfb） ----------
def viz_o3d_window(points, colors, poses, K, W, H):
    import open3d as o3d
    P = points; C = colors
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    if C is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(C,0,1))

    geoms = [pcd]
    # 轨迹
    traj = []
    for (R,t) in poses:
        Rwc = R.T; twc = -Rwc @ t
        traj.append(twc)
    if len(traj)>=2:
        traj = np.array(traj)
        lines = [(i,i+1) for i in range(len(traj)-1)]
        traj_ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(traj),
                                       lines=o3d.utility.Vector2iVector(lines))
        traj_ls.colors = o3d.utility.Vector3dVector([[0,0,1] for _ in lines])
        geoms.append(traj_ls)

    o3d.visualization.draw_geometries(geoms)  # 需要窗口系统

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mpl","o3d-offscreen","o3d"], default="o3d")
    ap.add_argument("--out", type=str, default="viz_test.png")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--points", type=int, default=60000, help="dummy点数（太大可调小）")
    args = ap.parse_args()

    # 数据
    P, C = make_dummy_pointcloud(n=args.points)
    poses = build_dummy_poses()
    K = pinhole_K(args.width, args.height, args.hfov)

    if args.backend == "mpl":
        viz_mpl(P, C, poses, K, args.width, args.height, args.out)

    elif args.backend == "o3d-offscreen":
        # 需 Open3D wheel 支持离屏(通常EGL)；若失败可退回 mpl 或用 xvfb + --backend o3d
        viz_o3d_offscreen(P, C, poses, K, args.width, args.height, args.out)

    else:  # "o3d"
        # 有桌面（或用 xvfb-run）时可交互查看
        viz_o3d_window(P, C, poses, K, args.width, args.height)

if __name__ == "__main__":
    main()
