# gs_render.py
# -*- coding: utf-8 -*-
"""
maintained by: Kaimin Mao(kaimin001@e.ntu.edu.sg)

使用 gsplat 渲染高斯溅射模型的工具函数

提供两个入口：
1) render_and_save_rt(...)   # 直接用 (R, t) (world->camera) 渲染并保存
2) render_and_save_Twc(...)  # 传入 T_wc（相机位姿，camera->world 的逆）自动换算后渲染并保存

"""

from typing import Iterable, List, Dict, Tuple, Union, Optional
import os
import numpy as np
import torch
import imageio

try:
    import gsplat as gs
except ImportError as e:
    raise ImportError(
        "找不到 gsplat，请先安装：pip install gsplat（或你环境里的对应包名）"
    ) from e


ArrayLike = Union[np.ndarray, torch.Tensor]
RT = Tuple[ArrayLike, ArrayLike]  # (R, t)


def _to_tensor(x: ArrayLike, device: str, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    x = np.asarray(x)
    return torch.tensor(x, device=device, dtype=dtype)


def _ensure_model_tensors(model: Dict[str, ArrayLike], device: str) -> Dict[str, torch.Tensor]:
    """
    期望的键：
      means:(N,3), quats:(N,4), scales:(N,3), opacities:(N,), colors:(N,3)
    接受 numpy 或 torch，统一转 torch.float32 并放到 device 上。
    """
    required = ["means", "quats", "scales", "opacities", "colors"]
    # 兼容命名 "opacs" -> "opacities"
    if "opacities" not in model and "opacs" in model:
        model["opacities"] = model["opacs"]

    missing = [k for k in required if k not in model]
    if missing:
        raise KeyError(f"model 缺少必要键：{missing}")

    out = {k: _to_tensor(model[k], device) for k in required}
    # 简单形状检查
    N = out["means"].shape[0]
    for k in required:
        if out[k].shape[0] != N:
            raise ValueError(f"model['{k}'] 的首维与 means 不一致：{out[k].shape} vs {out['means'].shape}")
    return out


def _make_K_tensor(intrinsics: Dict[str, ArrayLike], device: str) -> Tuple[torch.Tensor, int, int]:
    """
    intrinsics 支持两种方式：
      1) {"K": (3,3), "width": W, "height": H}
      2) {"fx":..., "fy":..., "cx":..., "cy":..., "width": W, "height": H}
    返回：K_torch(1,3,3), W, H
    """
    if "K" in intrinsics:
        K = _to_tensor(intrinsics["K"], device)
        if K.shape != (3, 3):
            raise ValueError(f"K 形状必须是 (3,3)，当前 {tuple(K.shape)}")
    else:
        for k in ["fx", "fy", "cx", "cy"]:
            if k not in intrinsics:
                raise KeyError(f"intrinsics 缺少键 '{k}' 或缺少 'K'")
        fx, fy, cx, cy = [float(intrinsics[k]) for k in ["fx", "fy", "cx", "cy"]]
        K = torch.tensor([[fx, 0.0, cx],
                          [0.0, fy, cy],
                          [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

    if "width" not in intrinsics or "height" not in intrinsics:
        raise KeyError("intrinsics 需要包含 'width' 和 'height'")

    W = int(intrinsics["width"])
    H = int(intrinsics["height"])

    # 扩一维，符合 gs.rasterization 的 (B,3,3)
    K = K.unsqueeze(0)
    return K, W, H


def _rt_to_viewmat(R: ArrayLike, t: ArrayLike, device: str) -> torch.Tensor:
    """
    (R, t) 为 world->camera。
    返回 viewmat (1,4,4) torch.float32。
    """
    R = _to_tensor(R, device)
    t = _to_tensor(t, device)
    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError(f"(R,t) 形状不对，R={tuple(R.shape)}, t={tuple(t.shape)}")
    T = torch.eye(4, dtype=torch.float32, device=device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T.unsqueeze(0)


def _twc_to_rt(T_wc: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    将相机位姿 T_wc（camera 在世界中的位姿）转为 (R_wc, t_wc)（world->camera）。
    约定：T_wc[:3,:3] 为 camera basis in world（OpenCV z-forward/x-right/y-down 列为基），
         C = T_wc[:3,3] 为相机中心（世界系）。
    则 world->camera:
         R_wc = R_int^T, t_wc = -R_wc @ C
    """
    T = np.asarray(T_wc, dtype=np.float32)
    if T.shape != (4, 4):
        raise ValueError(f"T_wc 形状必须是 (4,4)，当前 {tuple(T.shape)}")
    R_int = T[:3, :3]
    C = T[:3, 3]
    R_wc = R_int.T
    t_wc = - R_wc @ C
    return R_wc.astype(np.float32), t_wc.astype(np.float32)


@torch.no_grad()
def render_and_save_rt(
    poses_rt: Iterable[RT],
    img_paths: Iterable[str],
    model: Dict[str, ArrayLike],
    intrinsics: Dict[str, ArrayLike],
    device: str = "cuda",
    render_mode: str = "RGB",
    rasterize_mode: str = "antialiased",
    radius_clip: float = 0.0
) -> List[str]:
    """
    使用 (R,t)（world->camera）逐帧渲染并保存。

    参数：
        poses_rt:      [(R,t), ...]，R:(3,3)，t:(3,)
        img_paths:     与 poses 对齐的输出路径
        model:         dict，包含 means/quats/scales/opacities/colors
        intrinsics:    dict，见 _make_K_tensor() 说明
        device:        "cuda" 或 "cpu"
        render_mode:   传给 gs.rasterization（默认 "RGB"）
        rasterize_mode:"classic" 或 "antialiased"
        radius_clip:   半径裁剪

    返回：
        保存成功的文件路径列表（与输入 img_paths 顺序一致）
    """
    device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    mdl = _ensure_model_tensors(model, device)
    K_torch, W, H = _make_K_tensor(intrinsics, device)

    poses_list = list(poses_rt)
    paths_list = list(img_paths)
    if len(poses_list) != len(paths_list):
        raise ValueError("poses_rt and img_paths must have the same length")

    saved = []
    
    for idx, ((R, t), path) in enumerate(zip(poses_list, paths_list)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        viewmats = _rt_to_viewmat(R, t, device)  # (1,4,4)

        try:
            imgs, meta_img, meta = gs.rasterization(
                means=mdl["means"],      # (N,3)
                quats=mdl["quats"],      # (N,4) (w,x,y,z)
                scales=mdl["scales"],    # (N,3)
                opacities=mdl["opacities"],  # (N,)
                colors=mdl["colors"],    # (N,3)
                viewmats=viewmats,       # (1,4,4) world->camera
                Ks=K_torch,              # (1,3,3)
                width=W, height=H,
                render_mode=render_mode,
                rasterize_mode=rasterize_mode,
                radius_clip=radius_clip
            )
        except RuntimeError as err:
            raise RuntimeError(f"gs.rasterization failed at frame {idx} ({path}): {err}") from err

        rgb = imgs[0].clamp(0, 1).detach().cpu().numpy()  # (H,W,3)
        rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
        imageio.imwrite(path, rgb_u8)
        saved.append(path)
        print(f"[gs_render] Saved: {path}")

    return saved


@torch.no_grad()
def render_and_save_Twc(
    poses_Twc: Iterable[ArrayLike],
    img_paths: Iterable[str],
    model: Dict[str, ArrayLike],
    intrinsics: Dict[str, ArrayLike],
    device: str = "cuda",
    render_mode: str = "RGB",
    rasterize_mode: str = "classic",
    radius_clip: float = 0.0
) -> List[str]:
    """
    传入相机位姿 T_wc（camera 在世界中的位姿），内部换算成 (R,t) 后渲染并保存。
    其余参数同 render_and_save_rt。
    """
    poses_rt: List[RT] = []
        
    for T in poses_Twc:
        R_wc, t_wc = _twc_to_rt(T)
        poses_rt.append((R_wc, t_wc))
    return render_and_save_rt(
        poses_rt=poses_rt,
        img_paths=img_paths,
        model=model,
        intrinsics=intrinsics,
        device=device,
        render_mode=render_mode,
        rasterize_mode=rasterize_mode,
        radius_clip=radius_clip
    )
