#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
maintained by: Kaimin Mao(kaimin001@e.ntu.edu.sg)

Interactive Path & Camera Pose Planner for Gaussian Splat Models (Open3D GUI)
=============================================================================
Overview:
1) Load Gaussian centers (means) and optional scales as a point cloud for visualization.
2) Interactively choose a horizontal slicing plane z = const.
3) Click on the plane to create ordered control points; fit a smooth spline path.
4) Sample path points along the spline at a given spacing.
5) For each sample, generate a camera pose (T_wc, OpenCV convention z-forward/x-right/y-down),
   keep frames that satisfy:
   - Sufficient visible Gaussian centers; and
   - Overlap (Jaccard over visible sets) with the last kept frame.
6) Export intrinsics + poses to JSON (optionally convert later to (R,t) for renderers or SfM tools).

Usage (example):
    python manual_plane.py \
        --gaussians path/to/model_means_scales.npz \
        --fov 70 --imgw 1920 --imgh 1080 --near 0.1 --far 20.0 \
        --min_visible 800 --overlap_ratio 0.4 --ds 0.3

Accepted Gaussian inputs:
- .npz: keys 'means' (N,3) and optional 'scales' (N,3)
- .ply: points treated as 'means'; 'scales' filled with --default_scale

"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.interpolate import splprep, splev

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from tqdm import tqdm  
from render_from_plane import render_and_save_Twc
from render_depth import render_depth_and_save_Twc
import threading
import traceback
import sys

# ==========================
# Data structures & utilities
# ==========================

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @staticmethod
    def from_fov(w: int, h: int, fov_deg: float):
        # Assume horizontal FOV = fov_deg. Use square pixels => fy = fx
        fov = math.radians(fov_deg)
        fx = (w / 2.0) / math.tan(fov / 2.0)
        fy = fx
        cx, cy = w / 2.0, h / 2.0
        return CameraIntrinsics(fx, fy, cx, cy, w, h)

    def project(self, pts_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project camera-space points to pixels; return (uv, z)."""
        x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
        eps = 1e-8
        u = self.fx * (x / (z + eps)) + self.cx
        v = self.fy * (y / (z + eps)) + self.cy
        return np.stack([u, v], axis=-1), z


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-10:
        return v
    return v / n


def look_at(cam: np.ndarray, target: np.ndarray, up=np.array([0, 0, 1.0])) -> np.ndarray:
    """Return camera pose T_wc (4x4). OpenCV convention: z-forward, x-right, y-down.
    T_wc stores the camera center in world coords in T_wc[:3,3]. Columns are basis vectors.
    """
    forward = normalize(target - cam)
    right = normalize(np.cross(forward, up))
    true_up = normalize(np.cross(right, forward))
    R = np.stack([right, -true_up, forward], axis=1)  # columns are basis
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cam
    return T


def world_to_camera(T_wc: np.ndarray, pts_w: np.ndarray) -> np.ndarray:
    R = T_wc[:3, :3]
    t = T_wc[:3, 3]
    return (pts_w - t) @ R  # (N,3)


# ==========================
# Gaussian model loading & occupancy
# ==========================

def load_gaussians(path: str, default_scale=0.05):
    """
    支持 3D Gaussian Splatting 的 PLY：
      x,y,z, nx,ny,nz, f_dc_0..2, f_rest_*, opacity, scale_0..2, rot_0..3
    返回:
      means  (N,3) float32
      scales (N,3) float32
      colors (N,3) float32 in [0,1] 或 None
    """
    ext = os.path.splitext(path)[1].lower()
    colors = None

    if ext == ".npz":
        data = np.load(path)
        means = data["means"].astype(np.float32)
        scales = (data["scales"].astype(np.float32)
                  if "scales" in data else np.full_like(means, default_scale, dtype=np.float32))
        if "colors" in data:
            c = data["colors"].astype(np.float32)
            if c.max() > 1.5:  # 0-255 -> 0-1
                c = c / 255.0
            colors = c[:, :3]

    elif ext == ".ply":
        # Tensor API，能读取自定义属性
        try:
            tpcd = o3d.t.io.read_point_cloud(path)
            # 位置
            means = np.asarray(tpcd.point["positions"].numpy(), dtype=np.float32)

            # 尺度：3DGS 保存的是 log-scale，需要 exp 还原
            if all(k in tpcd.point for k in ["scale_0", "scale_1", "scale_2"]):
                s0 = tpcd.point["scale_0"].numpy().astype(np.float32)
                s1 = tpcd.point["scale_1"].numpy().astype(np.float32)
                s2 = tpcd.point["scale_2"].numpy().astype(np.float32)
                scales = np.exp(np.stack([s0, s1, s2], axis=1))  # 还原到正尺度
            else:
                scales = np.full_like(means, default_scale, dtype=np.float32)

            # 颜色：优先已有 colors；否则用 f_dc_0..2 的 sigmoid 近似基色
            if "colors" in tpcd.point and tpcd.point["colors"].shape[1] >= 3:
                c = tpcd.point["colors"].numpy().astype(np.float32)
                colors = np.clip(c[:, :3], 0.0, 1.0)
            elif all(k in tpcd.point for k in ["f_dc_0", "f_dc_1", "f_dc_2"]):
                f0 = tpcd.point["f_dc_0"].numpy().astype(np.float32)
                f1 = tpcd.point["f_dc_1"].numpy().astype(np.float32)
                f2 = tpcd.point["f_dc_2"].numpy().astype(np.float32)
                fdc = np.stack([f0, f1, f2], axis=1)
                # 常用做法：对 DC 项做 sigmoid，映射到 [0,1] 近似基色
                colors = 1.0 / (1.0 + np.exp(-fdc))
                colors = np.clip(colors, 0.0, 1.0)
            else:
                colors = None

        except Exception:
            # 回退到 legacy（无自定义属性，只有坐标）
            pcd = o3d.io.read_point_cloud(path)
            means = np.asarray(pcd.points, dtype=np.float32)
            scales = np.full_like(means, default_scale, dtype=np.float32)
            colors = (np.asarray(pcd.colors, dtype=np.float32)
                      if len(pcd.colors) == len(pcd.points) and len(pcd.colors) > 0 else None)
    else:
        raise ValueError(f"Unsupported gaussians file: {path}")

    return means, scales, colors


def plane_occupancy_2d(means: np.ndarray, scales: np.ndarray, z: float, thickness: float = 0.1,
                        grid_res: float = 0.1) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Project Gaussians near z into XY disks and mark occupancy grid (bool HxW)."""
    zmin, zmax = z - thickness / 2.0, z + thickness / 2.0
    mask = (means[:, 2] >= zmin) & (means[:, 2] <= zmax)
    pts = means[mask]
    if pts.shape[0] == 0:
        return np.zeros((1, 1), dtype=bool), (0, 0, 0, 0)
    xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
    xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
    pad = 1.0
    xmin -= pad; ymin -= pad; xmax += pad; ymax += pad

    W = max(1, int(math.ceil((xmax - xmin) / grid_res)))
    H = max(1, int(math.ceil((ymax - ymin) / grid_res)))

    occ = np.zeros((H, W), dtype=bool)
    rad = np.mean(scales[mask][:, :2], axis=1) * 2.0  # rough radius
    for (x, y), r in zip(pts[:, :2], rad):
        cx = int((x - xmin) / grid_res)
        cy = int((y - ymin) / grid_res)
        rr = max(1, int(r / grid_res))
        x0, x1 = max(0, cx - rr), min(W - 1, cx + rr)
        y0, y1 = max(0, cy - rr), min(H - 1, cy + rr)
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                dx = (xx - cx) * grid_res
                dy = (yy - cy) * grid_res
                if dx * dx + dy * dy <= r * r:
                    occ[yy, xx] = True
    return occ, (xmin, xmax, ymin, ymax)


# ==========================
# Spline fitting & sampling
# ==========================

def fit_spline_2d(xy: np.ndarray, smooth: float = 0.0) -> Tuple[object, float]:
    """Fit a 2D B-spline to points; return (tck, arc length)."""
    if xy.shape[0] < 2:
        raise ValueError("Need at least two points to fit a spline.")
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    s_norm = s / s[-1] if s[-1] > 0 else s
    tck, _ = splprep([xy[:, 0], xy[:, 1]], u=s_norm, k=min(3, xy.shape[0]-1), s=smooth)
    return tck, s[-1]


def sample_spline(tck, L: float, ds: float) -> np.ndarray:
    if L <= 0:
        return np.array([]).reshape(0, 2)
    n = max(2, int(math.ceil(L / ds)) + 1)
    u = np.linspace(0, 1, n)
    x, y = splev(u, tck)
    return np.stack([x, y], axis=1)


def tangent_on_spline(tck, u: float) -> np.ndarray:
    dx, dy = splev(u, tck, der=1)
    v = np.array([dx, dy])
    return normalize(v)


# ==========================
# Visibility & overlap (set-Jaccard)
# ==========================

def visible_points_mask(T_wc: np.ndarray, intr: CameraIntrinsics, pts_w: np.ndarray,
                        near: float, far: float, fov_h: float, fov_v: float) -> np.ndarray:
    """Simple visibility: depth in [near, far] and pixels inside image bounds."""
    pts_c = world_to_camera(T_wc, pts_w)
    uv, z = intr.project(pts_c)
    u, v = uv[:, 0], uv[:, 1]
    mask = (z > near) & (z < far) & (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    return mask


def overlap_ratio(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    a = set(np.where(idx_a)[0].tolist())
    b = set(np.where(idx_b)[0].tolist())
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(1, union)


# ==========================
# Open3D GUI App
# ==========================

class App:
    def __init__(self, args):
        self.args = args
        self.means, self.scales, self.colors = load_gaussians(args.gaussians, default_scale=args.default_scale)
        self.points = self.means  # only centers for planning
        self.intr = CameraIntrinsics.from_fov(args.imgw, args.imgh, args.fov)
        self.near = args.near
        self.far = args.far

        # GUI state
        self.plane_z = float(np.median(self.points[:, 2]))
        self.picked_xy: List[np.ndarray] = []
        self.tck = None
        self.curve_samples = None

        # Scene handles
        self.scene = None
        self.pcd = None
        self.plane = None
        self.window = None
        self.widget3d = None
        self.panel = None

        # Track temporary geometry names by group
        self._geom_names = {
            "picks": set(),
            "samples": set(),
            "aux": set(),
        }

        self.click_generation_num = 0
        self._build_ui()

    # ---------- Geometry name bookkeeping ----------
    def _add_geom(self, name, geom, material=None, group=None):
        if material is None:
            material = rendering.MaterialRecord()
        if self.scene.has_geometry(name):
            self.scene.remove_geometry(name)
        self.scene.add_geometry(name, geom, material)
        if group is not None:
            self._geom_names[group].add(name)

    def _remove_geom(self, name, group=None):
        if self.scene.has_geometry(name):
            self.scene.remove_geometry(name)
        if group is not None:
            self._geom_names[group].discard(name)

    def _clear_group(self, group):
        for name in list(self._geom_names[group]):
            self._remove_geom(name, group)

    # ---------- UI ----------
    def _build_ui(self):
        app = gui.Application.instance
        app.initialize()

        self.window = gui.Application.instance.create_window("Gaussian Path Planner", 1280, 800)
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.set_on_mouse(self._on_mouse)
        self.window.add_child(self.widget3d)

        em = self.window.theme.font_size
        margin = 0.5 * em

        panel = gui.Vert(0, gui.Margins(margin, margin, margin, margin))
        self.panel = panel

        # Plane Z
        self.z_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.z_edit.set_value(self.plane_z)
        self.z_edit.set_on_value_changed(self._on_z_changed)
        panel.add_child(gui.Label("Slice plane z"))
        panel.add_child(self.z_edit)

        # Plane button
        btn_plane = gui.Button("Update / Show plane")
        btn_plane.set_on_clicked(self._update_plane)
        panel.add_child(btn_plane)

        # Picking help
        panel.add_child(gui.Label("Click on plane to add path points (in order)"))
        btn_undo = gui.Button("Undo last point")
        btn_undo.set_on_clicked(self._undo_point)
        panel.add_child(btn_undo)
        btn_clear = gui.Button("Clear points")
        btn_clear.set_on_clicked(self._clear_points)
        panel.add_child(btn_clear)

        # Sampling params
        self.ds_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.ds_edit.set_value(self.args.ds)
        panel.add_child(gui.Label("Path sample spacing ds (m)"))
        panel.add_child(self.ds_edit)

        self.overlap_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.overlap_edit.set_value(self.args.overlap_ratio)
        panel.add_child(gui.Label("Overlap threshold [0,1]"))
        panel.add_child(self.overlap_edit)

        self.min_visible_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self.min_visible_edit.set_value(self.args.min_visible)
        panel.add_child(gui.Label("Min visible Gaussian centers per frame"))
        panel.add_child(self.min_visible_edit)

        # Actions
        btn_fit = gui.Button("Fit spline & sample")
        btn_fit.set_on_clicked(self._fit_and_sample)
        panel.add_child(btn_fit)

        btn_pose = gui.Button("Generate poses & export JSON & render images")
        btn_pose.set_on_clicked(self._gen_and_export_poses)
        panel.add_child(btn_pose)

        # Guidance (consolidated text to avoid overlap)
        guide_text = (
            "How to use:\n"
            "1) Hold Shift + Left Click on the green plane to add points (in order).\n"
            "2) Click 'Fit spline & sample' to generate samples.\n"
            "3) Click 'Generate poses...' to export JSON and render images."
        )
        panel.add_child(gui.Label(guide_text))

        # Busy/status indicator (blank until used)
        self._busy_label = gui.Label("")
        panel.add_child(self._busy_label)

        # Layout & init scene
        self.window.add_child(panel)
        self.window.set_on_layout(self._on_layout)
        self._init_scene()

        app.run()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 320
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)
        self.widget3d.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)

    # ---------- Scene init ----------
    def _init_scene(self):
        self.scene = self.widget3d.scene
        self.scene.set_background([0.05, 0.05, 0.06, 1.0])

        # Points
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points)

        if getattr(self, "colors", None) is not None and len(self.colors) == self.points.shape[0]:
            # 直接使用文件自带颜色
            c = np.clip(self.colors, 0.0, 1.0)
        else:
            # 没有颜色 → 按 z 高度着色（蓝→红）
            z = self.points[:, 2]
            z_min, z_max = float(np.min(z)), float(np.max(z))
            denom = (z_max - z_min) if (z_max > z_min) else 1.0
            t = ((z - z_min) / denom).reshape(-1, 1)  # [0,1]
            # 简单蓝红渐变：b=1-t, r=t, g=中间过渡
            r = t
            g = 0.5 * (1.0 - np.abs(2.0 * t - 1.0))
            b = 1.0 - t
            c = np.concatenate([r, g, b], axis=1).astype(np.float32)

        self.pcd.colors = o3d.utility.Vector3dVector(c)
        
        # 添加到场景
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"   # 或 "defaultLit"
        mat.point_size = 3.0          # 调大点大小，便于观察（可再调大）
        self._add_geom("points", self.pcd, mat)

        
        # Plane
        self._update_plane()

        # Camera
        aabb = self.scene.bounding_box
        self.widget3d.setup_camera(60.0, aabb, aabb.get_center())

    def _update_plane(self):
        z = float(self.z_edit.double_value)
        self.plane_z = z

        if self.scene.has_geometry("plane"):
            self.scene.remove_geometry("plane")

        aabb = self.pcd.get_axis_aligned_bounding_box()
        size = max(aabb.get_extent()) * 1.2
        depth = 1e-3

        plane = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=depth)
        plane.compute_vertex_normals()

        center = aabb.get_center()
        T = np.eye(4)
        T[:3, 3] = [center[0] - size / 2,
                    center[1] - size / 2,
                    z - depth * 0.5]  # 平面中心对齐 z
        plane.transform(T)
        plane.paint_uniform_color([0.2, 0.6, 0.2])
        plane.compute_vertex_normals()
        self.plane = plane

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = (0.2, 0.3, 0.75, 0.5)
        self._add_geom("plane", self.plane, mat)

        # 保留或移除都行：更新平面时是否清空选点
        self._clear_points()

    def _on_z_changed(self, val):
        # Only update value; geometry updates when pressing the button
        pass

    def _undo_point(self):
        if len(self.picked_xy) > 0:
            self.picked_xy.pop()
            self._refresh_polyline_and_markers()

    def _clear_points(self):
        self.picked_xy.clear()
        if self.scene.has_geometry("polyline"):
            self.scene.remove_geometry("polyline")
        self._clear_group("picks")
        self._clear_group("samples")

    def _refresh_polyline_and_markers(self):
        if self.scene.has_geometry("polyline"):
            self.scene.remove_geometry("polyline")

        if len(self.picked_xy) >= 2:
            # Slightly raise the polyline above the plane to avoid z-fighting
            z_eps = 1e-3
            pts3d = np.array([[x, y, self.plane_z + z_eps] for x, y in self.picked_xy])
            lines = np.stack([np.arange(len(pts3d)-1), np.arange(1, len(pts3d))], axis=1)
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(pts3d),
                lines=o3d.utility.Vector2iVector(lines)
            )
            colors = np.tile(np.array([[1.0, 0.6, 0.0]]), (len(lines), 1))
            line_set.colors = o3d.utility.Vector3dVector(colors)
            mat = rendering.MaterialRecord()
            mat.shader = "unlitLine"
            try:
                mat.line_width = 4.0
            except Exception:
                pass
            self._add_geom("polyline", line_set, mat)

        # Rebuild pick spheres
        self._clear_group("picks")
        for i, (x, y) in enumerate(self.picked_xy):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            s.translate([x, y, self.plane_z])
            s.paint_uniform_color([1.0, 0.0, 0.0])
            name = f"pick_{i:03d}"
            self._add_geom(name, s, rendering.MaterialRecord(), group="picks")

    # ---------- Ray from mouse (no camera.create_ray needed) ----------
    def _ray_from_mouse(self, x, y):
        frame = self.widget3d.frame
        ndc_x = (x - frame.x) / max(1e-6, frame.width) * 2.0 - 1.0
        ndc_y = 1.0 - (y - frame.y) / max(1e-6, frame.height) * 2.0
        cam = self.widget3d.scene.camera
        P = np.asarray(cam.get_projection_matrix(), dtype=np.float64)
        V = np.asarray(cam.get_view_matrix(), dtype=np.float64)
        PV = P @ V
        invPV = np.linalg.inv(PV)
        vnear = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float64)
        vfar  = np.array([ndc_x, ndc_y,  1.0, 1.0], dtype=np.float64)
        near_w = invPV @ vnear; near_w = near_w[:3] / max(1e-12, near_w[3])
        far_w  = invPV @ vfar;  far_w  = far_w[:3]  / max(1e-12, far_w[3])
        origin = near_w
        direction = far_w - near_w
        n = np.linalg.norm(direction)
        if n < 1e-12:
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            direction = direction / n
        return origin.astype(np.float64), direction.astype(np.float64)

    # ---------- Mouse interaction: click on plane to add points ----------
# === 替换整个 _on_mouse 函数为以下内容 ===
    def _on_mouse(self, event: gui.MouseEvent):
        # 仅当按住 Shift + 左键“按下”时进行拾取；其余交给默认交互（旋转/平移/缩放）
        # --- 跨版本检测左键按下 ---
        is_left_down = False
        try:
            is_left_down = (event.type == gui.MouseEvent.Type.BUTTON_DOWN and
                            event.is_button_down(gui.MouseButton.LEFT))
        except AttributeError:
            try:
                is_left_down = (event.type == gui.MouseEvent.Type.BUTTON_DOWN and
                                getattr(event, "button", None) == gui.MouseButton.LEFT)
            except Exception:
                try:
                    btns = int(getattr(event, "buttons", 0))
                    is_left_down = (event.type == gui.MouseEvent.Type.BUTTON_DOWN and
                                    (btns & int(gui.MouseButton.LEFT)) != 0)
                except Exception:
                    is_left_down = False

        # 需要按住 Shift 才触发拾取
        # --- 兼容不同版本：把 modifiers 和 KeyModifier 都转成 int 再做位运算 ---
        mods = getattr(event, "modifiers", 0)
        try:
            mods_int = int(mods)
        except Exception:
            mods_int = 0
        try:
            shift_mask = int(gui.KeyModifier.SHIFT)
        except Exception:
            shift_mask = 1 << 0  # 兜底

        shift_held = (mods_int & shift_mask) != 0

        if is_left_down and shift_held:
            # 用投影/视图矩阵反投影成射线（兼容 0.19，无需 camera.create_ray）
            origin, direction = self._ray_from_mouse(event.x, event.y)
            if abs(direction[2]) < 1e-12:
                return gui.Widget.EventCallbackResult.IGNORED  # 不阻断默认交互
            t = (self.plane_z - origin[2]) / direction[2]
            if t <= 0:
                return gui.Widget.EventCallbackResult.IGNORED
            hit = origin + t * direction
            x, y = float(hit[0]), float(hit[1])

            self.picked_xy.append(np.array([x, y]))
            self._refresh_polyline_and_markers()
            return gui.Widget.EventCallbackResult.CONSUMED

        # 其余所有事件（包含未按 Shift 的左键、右键、中键、拖拽、滚轮）全部交给 Open3D 默认相机控制
        return gui.Widget.EventCallbackResult.IGNORED

    def _notify(self, title: str, text: str):
        """Cross-version notification: try MessageBox -> simple Dialog -> print."""
        try:
            # Open3D 常见版本
            mb = gui.MessageBox(title, text)  # 有的版本叫 MessageBox
            self.window.show_dialog(mb)
            return
        except Exception:
            pass
        try:
            # 另一种命名（若存在）
            md = gui.MessageDialog(title, text)  # 某些构建可能是 MessageDialog
            self.window.show_dialog(md)
            return
        except Exception:
            pass
        try:
            # 兜底：自己拼一个最简单的对话框
            dlg = gui.Dialog(title)
            em = self.window.theme.font_size
            margin = int(round(0.5 * em))
            layout = gui.Vert(margin, gui.Margins(margin, margin, margin, margin))
            layout.add_child(gui.Label(text))
            ok = gui.Button("OK")
            def _close():
                try:
                    self.window.close_dialog()
                except Exception:
                    pass
            ok.set_on_clicked(_close)
            layout.add_child(ok)
            dlg.add_child(layout)
            self.window.show_dialog(dlg)
            return
        except Exception:
            pass
        # 最后实在不行就打印
        print(f"[{title}] {text}")

    # ---------- Fit & sample ----------
    def _fit_and_sample(self):
        if len(self.picked_xy) < 2:
            self._notify("Hint", "Please click at least two points to generate a path.")
            return
        xy = np.stack(self.picked_xy, axis=0)
        self.tck, L = fit_spline_2d(xy, smooth=0.0)
        ds = float(self.ds_edit.double_value)
        samples_xy = sample_spline(self.tck, L, ds)
        self.curve_samples = np.concatenate([samples_xy, np.full((samples_xy.shape[0], 1), self.plane_z)], axis=1)

        # Visualize samples
        self._clear_group("samples")
        for i, p in enumerate(self.curve_samples):
            sp = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
            sp.translate(p)
            sp.paint_uniform_color([0.0, 0.6, 1.0])
            name = f"sample_{i:03d}"
            self._add_geom(name, sp, rendering.MaterialRecord(), group="samples")

        self._notify("Done", f"Spline length ≈ {L:.2f} m, sampled {len(self.curve_samples)} points.")

    # ---------- Pose generation & export & render image ----------
    def _gen_and_export_poses(self):
        # Run heavy generation + rendering in a background thread to avoid UI freeze
        if getattr(self, "_is_busy", False):
            self._notify("Busy", "Processing is already running.")
            return
        if self.curve_samples is None or self.tck is None:
            self._notify("Hint", "Please fit spline and sample first.")
            return

        def worker():
            try:
                self.click_generation_num += 1

                # Update UI label to indicate work started
                def _set_busy(msg: str):
                    try:
                        self._busy_label.text = msg
                    except Exception:
                        pass
                gui.Application.instance.post_to_main_thread(self.window, lambda: _set_busy("Working... Generating poses and images..."))

                pts_w = self.points
                center_all = np.mean(pts_w, axis=0)

                n = self.curve_samples.shape[0]
                us = np.linspace(0, 1, n)
                tangents = np.array([tangent_on_spline(self.tck, u) for u in us])

                poses = []
                vis_masks = []
                for i, (p, t2) in enumerate(zip(self.curve_samples, tangents)):
                    fwd = normalize(t2)
                    look = p + np.array([fwd[0], fwd[1], 0.0]) * 2.0
                    look = 0.7 * look + 0.3 * center_all
                    look[2] = look[2] - 0.2

                    T_wc = look_at(p, look, up=np.array([0, 0, 1.0]))
                    mask = visible_points_mask(
                        T_wc, self.intr, pts_w, self.near, self.far,
                        fov_h=self.args.fov,
                        fov_v=self.args.fov * (self.args.imgh / self.args.imgw)
                    )
                    poses.append(T_wc)
                    vis_masks.append(mask)

                min_visible = int(self.min_visible_edit.int_value)
                min_overlap = float(self.overlap_edit.double_value)
                keep = []
                last_kept = -1
                for i in range(n):
                    if vis_masks[i].sum() < min_visible:
                        continue
                    if last_kept >= 0:
                        ov = overlap_ratio(vis_masks[last_kept], vis_masks[i])
                        if ov < min_overlap:
                            continue
                    keep.append(i)
                    last_kept = i

                kept_poses = [poses[i] for i in keep]
                kept_ids = keep

                print("Generating rendered images...")
                sys.stdout.flush()
                def _set_busy(msg: str):
                    try:
                        self._busy_label.text = msg
                    except Exception:
                        pass
                gui.Application.instance.post_to_main_thread(self.window, lambda: _set_busy("Rendering RGB 0/%d" % (len(kept_poses),)))

                os.makedirs(self.args.img_paths, exist_ok=True)
                os.makedirs(self.args.depth_paths, exist_ok=True)

                from load_ply import load_gaussians_from_ply
                scene = load_gaussians_from_ply(self.args.gaussians, device="cuda")
                for k in scene:
                    scene[k] = scene[k].to("cuda").contiguous().float()

                import torch
                opacs  = torch.sigmoid(scene['opacity'].squeeze(-1)) 
                scales = torch.exp(scene['scale'])

                # scene['rot'] shape: (N,4)
                q = scene['rot'].contiguous().float()  # 原始四元数

                # ----- 自检顺序（启发式）：如果第4列均值绝对值最大，说明可能是 xyzw，需要重排到 wxyz -----
                with torch.no_grad():
                    col_abs_mean = q.abs().mean(dim=0)           # (4,)
                    likely_xyzw = col_abs_mean[-1] > col_abs_mean[:-1].max()  # 第4列最大 → 怀疑是 w 在最后
                if likely_xyzw:
                    q = q[:, [3, 0, 1, 2]]   # xyzw -> wxyz

                # 必须归一化（防数值漂移）
                q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)

                model = {
                    "means": scene['xyz'],         # (N,3)
                    "quats": q,        # (N,4)
                    "scales": scales,       # (N,3)
                    "opacities": opacs,     # (N,)
                    "colors": scene['rgb']        # (N,3) in [0,1]
                }

                HFOV = math.radians(self.args.fov)
                fx = fy = self.args.imgw / (2*math.tan(HFOV/2))
                cx, cy = self.args.imgw/2.0, self.args.imgh/2.0

                # 用 numpy K（render_* 接口会再转 tensor），无需先放 GPU
                K = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]], dtype=np.float32)
                intr = {"K": K, "width": self.args.imgw, "height": self.args.imgh}
                # Build output paths
                img_dir = os.path.join(self.args.img_paths, str(self.click_generation_num))
                depth_dir = os.path.join(self.args.depth_paths, str(self.click_generation_num))
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(depth_dir, exist_ok=True)
                img_paths = [os.path.join(img_dir, f"frame_{i:04d}.png") for i in range(len(kept_poses))]
                depth_paths = [os.path.join(depth_dir, f"frame_{i:04d}.png") for i in range(len(kept_poses))]

                # Render RGB strictly per-frame with real-time progress updates
                rgb_success_ids = []
                for idx, (pose, path) in enumerate(zip(kept_poses, img_paths)):
                    gui.Application.instance.post_to_main_thread(
                        self.window,
                        lambda i=idx, N=len(kept_poses): _set_busy(f"Rendering RGB {i+1}/{N}")
                    )
                    try:
                        render_and_save_Twc([pose], [path], model, intr, device="cuda")
                        rgb_success_ids.append(idx)
                    except Exception as _e:
                        print(f"[RGB] failed at {idx}: {_e}")
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass

                # Render Depth strictly per-frame for frames with successful RGB
                depth_success_ids = []
                for j, i in enumerate(rgb_success_ids):
                    gui.Application.instance.post_to_main_thread(
                        self.window,
                        lambda k=j, N=len(rgb_success_ids): _set_busy(f"Rendering Depth {k+1}/{N}")
                    )
                    try:
                        render_depth_and_save_Twc([kept_poses[i]], [depth_paths[i]], model, intr, device="cuda", depth_scale=1000.0)
                        depth_success_ids.append(i)
                    except Exception as _e:
                        print(f"[Depth] failed at {i}: {_e}")
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass

                # Final set of frames we consider successfully rendered (use RGB success as ground truth)
                final_ids = rgb_success_ids

                # Compose JSON only with successfully rendered frames to guarantee 1:1
                out = {
                    "intrinsics": {
                        "fx": self.intr.fx,
                        "fy": self.intr.fy,
                        "cx": self.intr.cx,
                        "cy": self.intr.cy,
                        "width": self.intr.width,
                        "height": self.intr.height,
                        "fov_deg": self.args.fov,
                        "near": self.near,
                        "far": self.far,
                    },
                    "poses": [
                        {
                            "id": int(kept_ids[i]),
                            "T_wc": np.asarray(kept_poses[i]).reshape(4, 4).tolist(),
                            "position": np.asarray(kept_poses[i])[:3, 3].tolist(),
                        }
                        for i in final_ids
                    ],
                    "meta": {
                        "total_samples": int(n),
                        "kept": len(kept_ids),
                        "rendered_rgb": len(final_ids),
                        "rendered_depth": int(len(set(final_ids) & set(depth_success_ids))),
                        "min_visible": int(min_visible),
                        "min_overlap": float(min_overlap),
                        "ds": float(self.ds_edit.double_value),
                        "click_generation": int(self.click_generation_num),
                    }
                }
                os.makedirs(self.args.pose_outdir, exist_ok=True)
                json_path = os.path.join(self.args.pose_outdir, f"planned_poses_{self.click_generation_num}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)

                # Cleanup pass: remove unexpected files not listed as successful
                removed_rgb = 0
                removed_depth = 0
                try:
                    keep_rgb = set(final_ids)
                    for fname in os.listdir(img_dir):
                        if not (fname.startswith("frame_") and fname.endswith(".png")):
                            continue
                        try:
                            idx = int(os.path.splitext(fname)[0].split("_")[1])
                        except Exception:
                            continue
                        if idx not in keep_rgb:
                            try:
                                os.remove(os.path.join(img_dir, fname))
                                removed_rgb += 1
                            except Exception:
                                pass
                except Exception:
                    pass

                try:
                    keep_depth = set(depth_success_ids)
                    for fname in os.listdir(depth_dir):
                        if not (fname.startswith("frame_") and fname.endswith(".png")):
                            continue
                        try:
                            idx = int(os.path.splitext(fname)[0].split("_")[1])
                        except Exception:
                            continue
                        if idx not in keep_depth:
                            try:
                                os.remove(os.path.join(depth_dir, fname))
                                removed_depth += 1
                            except Exception:
                                pass
                except Exception:
                    pass

                summary = (
                    f"Kept frames: {len(kept_ids)}/{n}. "
                    f"Rendered RGB: {len(final_ids)}/{len(kept_ids)}. "
                    f"Rendered Depth: {len(set(final_ids) & set(depth_success_ids))}/{len(final_ids)}.\n"
                    f"Cleaned extras -> RGB: {removed_rgb}, Depth: {removed_depth}.\n"
                    f"JSON: {json_path}\n"
                    f"RGB dir: {img_dir}\nDepth dir: {depth_dir}"
                )
                gui.Application.instance.post_to_main_thread(self.window, lambda: self._notify("Done", summary))

            except Exception as e:
                tb = traceback.format_exc()
                print(tb, file=sys.stderr)
                sys.stderr.flush()
                _msg = str(e)
                gui.Application.instance.post_to_main_thread(self.window, lambda m=_msg: self._notify("Error", m))
            finally:
                def _clear_busy():
                    try:
                        self._busy_label.text = ""
                        self._is_busy = False
                    except Exception:
                        pass
                gui.Application.instance.post_to_main_thread(self.window, _clear_busy)

        self._is_busy = True
        threading.Thread(target=worker, daemon=True).start()

# ==========================
# Entry
# ==========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussians", type=str, required=True, help=".npz with means(,scales) or .ply as centers")
    parser.add_argument("--pose_outdir", type=str, default="./manual_out_poses")
    parser.add_argument("--img_paths", type=str, default="./manual_out_img", help="Path to the output images")
    parser.add_argument("--depth_paths", type=str, default="./manual_out_depth", help="Path to the output depth maps")
    parser.add_argument("--default_scale", type=float, default=0.05, help="Default Gaussian approx scale for .ply")

    # Camera / frustum
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--imgw", type=int, default=1920)
    parser.add_argument("--imgh", type=int, default=1080)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=20.0)

    # Path sampling & filtering
    parser.add_argument("--ds", type=float, default=0.3, help="Path sampling spacing")
    parser.add_argument("--overlap_ratio", type=float, default=0.4, help="Jaccard overlap threshold [0,1]")
    parser.add_argument("--min_visible", type=int, default=800, help="Min visible centers per frame")
    
    args = parser.parse_args()

    App(args)


if __name__ == "__main__":
    main()
