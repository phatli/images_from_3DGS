import numpy as np
import torch
from plyfile import PlyData

def list_fields(verts):
    names = verts.data.dtype.names
    print("[PLY fields]", names)
    return set(names)

def stack_fields(verts, names):
    return np.vstack([verts[n] for n in names]).T

def extract_xyz(verts, fields):
    # 常见命名：x,y,z
    need = ['x','y','z']
    if all(n in fields for n in need):
        xyz = stack_fields(verts, need)
        return torch.tensor(xyz, dtype=torch.float32)
    raise ValueError("未找到坐标字段 x,y,z。现有字段: {}".format(sorted(fields)))

def extract_rgb(verts, fields):
    # 1) 直接RGB（uint8或float）
    if all(n in fields for n in ['r','g','b']):
        rgb = stack_fields(verts, ['r','g','b']).astype(np.float32)
        if rgb.max() > 1.001:  # 0-255 -> 0-1
            rgb = rgb / 255.0
        return torch.tensor(rgb, dtype=torch.float32)

    if all(n in fields for n in ['red','green','blue']):
        rgb = stack_fields(verts, ['red','green','blue']).astype(np.float32)
        if rgb.max() > 1.001:
            rgb = rgb / 255.0
        return torch.tensor(rgb, dtype=torch.float32)

    # 2) color_0..2 / colors_0..2
    for base in ['color', 'colors']:
        cand = [f"{base}_0", f"{base}_1", f"{base}_2"]
        if all(n in fields for n in cand):
            rgb = stack_fields(verts, cand).astype(np.float32)
            if rgb.max() > 1.001:
                rgb = rgb / 255.0
            return torch.tensor(rgb, dtype=torch.float32)

    # 3) SH DC（Graphdeco/其他导出器）
    for base in ['f_dc', 'sh_dc']:
        cand = [f"{base}_0", f"{base}_1", f"{base}_2"]
        if all(n in fields for n in cand):
            dc = stack_fields(verts, cand).astype(np.float32)
            # 经验做法：用 sigmoid 把 DC 系数挤回 [0,1] 作为近似颜色
            rgb = 1.0 / (1.0 + np.exp(-dc))
            return torch.tensor(rgb, dtype=torch.float32)

    raise ValueError("未找到颜色字段（r,g,b / red,green,blue / color_0..2 / f_dc_0..2 / sh_dc_0..2）。现有字段: {}".format(sorted(fields)))

def extract_opacity(verts, fields):
    for name in ['opacity', 'alpha', 'a']:
        if name in fields:
            opa = verts[name].astype(np.float32)
            # 若数值异常，可裁剪
            return torch.tensor(opa).unsqueeze(1)
    # 没有就默认全不透明
    return torch.ones((len(verts), 1), dtype=torch.float32)

def extract_scale(verts, fields):
    # 常见：scale_0..2；有的存单通道 scale、或 log_scale_*，自己按需 exp
    if all(n in fields for n in ['scale_0','scale_1','scale_2']):
        arr = stack_fields(verts, ['scale_0','scale_1','scale_2']).astype(np.float32)
        return torch.tensor(arr, dtype=torch.float32)
    if 'scale' in fields:
        s = verts['scale'].astype(np.float32)
        if s.ndim == 1:
            s = np.stack([s, s, s], axis=1)
        return torch.tensor(s, dtype=torch.float32)
    # 有些导出器是 log_scale_*
    if all(n in fields for n in ['log_scale_0','log_scale_1','log_scale_2']):
        arr = stack_fields(verts, ['log_scale_0','log_scale_1','log_scale_2']).astype(np.float32)
        arr = np.exp(arr)
        return torch.tensor(arr, dtype=torch.float32)
    raise ValueError("未找到 scale 字段（scale_0..2/scale/log_scale_0..2）。现有字段: {}".format(sorted(fields)))

def extract_rot_quat(verts, fields):
    # 常见：rot_0..3（多数是 x,y, z, w 或 w,x,y,z，需按你的渲染器要求调整）
    candidates = [
        ['rot_0','rot_1','rot_2','rot_3'],
        ['q_x','q_y','q_z','q_w'],
        ['quat_x','quat_y','quat_z','quat_w'],
        ['qx','qy','qz','qw'],
        ['qw','qx','qy','qz'],
    ]
    for cand in candidates:
        if all(n in fields for n in cand):
            q = stack_fields(verts, cand).astype(np.float32)
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
            return torch.tensor(q, dtype=torch.float32)
    raise ValueError("未找到四元数字段（rot_* 或 q*）。现有字段: {}".format(sorted(fields)))

def load_gaussians_from_ply(ply_path, device="cuda"):
    ply = PlyData.read(ply_path)
    verts = ply['vertex']
    fields = list_fields(verts)

    xyz = extract_xyz(verts, fields).to(device)
    rgb = extract_rgb(verts, fields).to(device)
    opacity = extract_opacity(verts, fields).to(device)
    scale = extract_scale(verts, fields).to(device)
    rot = extract_rot_quat(verts, fields).to(device)

    print("xyz:", xyz.shape, xyz.dtype)
    print("rgb:", rgb.shape, rgb.dtype, "min/max:", rgb.min().item(), rgb.max().item())
    print("opacity:", opacity.shape, opacity.dtype, "min/max:", opacity.min().item(), opacity.max().item())
    print("scale:", scale.shape, scale.dtype)
    print("rot (quat):", rot.shape, rot.dtype)

    return dict(xyz=xyz, rgb=rgb, opacity=opacity, scale=scale, rot=rot)