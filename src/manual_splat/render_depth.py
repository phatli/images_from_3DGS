# depth_render.py
# -*- coding: utf-8 -*-

"""
按位姿渲染深度图并保存的工具函数（基于 gsplat）
maintained by: Kaimin Mao(kaimin001@e.ntu.edu.sg)

提供：
- render_depth_and_save_rt(poses_rt, img_paths, model, intrinsics, ...)
- render_depth_and_save_Twc(poses_Twc, img_paths, model, intrinsics, ...)

说明：
- model: dict，包含 {"means","quats","scales","opacities","colors"}，numpy 或 torch 均可
- intrinsics: 传 {"K":(3,3),"width":W,"height":H} 或 {"fx","fy","cx","cy","width","height"}
- img_paths: 每帧对应一个输出路径；根据 depth_format 保存为 .png/.npy

保存格式：
- depth_format = "uint16_png"  -> 16 位 PNG，单位毫米（depth_scale=1000.0）
- depth_format = "float32_npy" -> .npy，单位米（float32）
"""

from typing import Iterable, List, Dict, Tuple, Union
import os
import numpy as np
import torch
import imageio

try:
    import gsplat as gs
except ImportError as e:
    raise ImportError(
        "未找到 gsplat，请先安装：pip install gsplat（或你的环境对应包名）"
    ) from e

ArrayLike = Union[np.ndarray, torch.Tensor]
RT = Tuple[ArrayLike, ArrayLike]  # (R, t)


# ------------------ 工具函数 ------------------
# 
def _debug_print_render_outputs(imgs, meta_img, meta, tag=""):
    import torch as _torch

    def _shape(x):
        if isinstance(x, _torch.Tensor):
            return f"Tensor{tuple(x.shape)} dtype={x.dtype}"
        if isinstance(x, (list, tuple)):
            return f"{type(x).__name__}[len={len(x)}]"
        if isinstance(x, dict):
            # 只展示前若干个 key，避免刷屏
            ks = list(x.keys())
            return f"dict(keys={ks[:20]})"
        return type(x).__name__

    print(f"[debug]{tag} imgs:     {_shape(imgs)}")
    print(f"[debug]{tag} meta_img: {_shape(meta_img)}")
    if isinstance(meta, dict):
        print(f"[debug]{tag} meta keys: {list(meta.keys())}")
        for k, v in meta.items():
            if isinstance(v, _torch.Tensor):
                print(f"[debug]{tag}  - {k}: Tensor{tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"[debug]{tag}  - {k}: {type(v).__name__}")
    else:
        print(f"[debug]{tag} meta: {type(meta).__name__}")


def _to_tensor(x: ArrayLike, device: str, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    x = np.asarray(x)
    return torch.tensor(x, device=device, dtype=dtype)


def _ensure_model_tensors(model: Dict[str, ArrayLike], device: str) -> Dict[str, torch.Tensor]:
    """
    期望键：
      means:(N,3), quats:(N,4), scales:(N,3), opacities:(N,), colors:(N,3)
    """
    required = ["means", "quats", "scales", "opacities", "colors"]
    if "opacities" not in model and "opacs" in model:
        model["opacities"] = model["opacs"]
    missing = [k for k in required if k not in model]
    if missing:
        raise KeyError(f"model 缺少必要键：{missing}")

    out = {k: _to_tensor(model[k], device) for k in required}
    N = out["means"].shape[0]
    for k in required:
        if out[k].shape[0] != N:
            raise ValueError(f"model['{k}'] 的首维与 means 不一致：{out[k].shape} vs {out['means'].shape}")
    return out


def _make_K_tensor(intrinsics: Dict[str, ArrayLike], device: str) -> Tuple[torch.Tensor, int, int]:
    """
    intrinsics:
      - {"K": (3,3), "width": W, "height": H}
      - 或 {"fx","fy","cx","cy","width","height"}
    返回：K(1,3,3), W, H
    """
    if "K" in intrinsics:
        K = _to_tensor(intrinsics["K"], device)
        if K.shape != (3, 3):
            raise ValueError(f"K 形状必须是 (3,3)，当前 {tuple(K.shape)}")
    else:
        for k in ["fx", "fy", "cx", "cy"]:
            if k not in intrinsics:
                raise KeyError(f"intrinsics 缺少 '{k}' 或缺少 'K'")
        fx, fy, cx, cy = [float(intrinsics[k]) for k in ["fx", "fy", "cx", "cy"]]
        K = torch.tensor([[fx, 0.0, cx],
                          [0.0, fy, cy],
                          [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    if "width" not in intrinsics or "height" not in intrinsics:
        raise KeyError("intrinsics 需要包含 'width' 和 'height'")
    W = int(intrinsics["width"])
    H = int(intrinsics["height"])
    return K.unsqueeze(0), W, H


def _rt_to_viewmat(R: ArrayLike, t: ArrayLike, device: str) -> torch.Tensor:
    """
    (R, t) 为 world->camera。返回 viewmat (1,4,4)。
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
    T_wc: camera 在世界中的位姿（Camera->World 的逆）
    返回 world->camera 的 (R_wc, t_wc)。
    """
    T = np.asarray(T_wc, dtype=np.float32)
    if T.shape != (4, 4):
        raise ValueError(f"T_wc 形状必须是 (4,4)，当前 {tuple(T.shape)}")
    R_int = T[:3, :3]   # camera basis in world
    C = T[:3, 3]        # camera center in world
    R_wc = R_int.T
    t_wc = - R_wc @ C
    return R_wc.astype(np.float32), t_wc.astype(np.float32)


def _extract_depth(imgs, meta_img, meta, prefer_key: str = None, allow_meta_img: bool = True) -> torch.Tensor:

    # A) 直接从 imgs 取
    if isinstance(imgs, torch.Tensor):
        if imgs.dim() == 3 and imgs.shape[0] == 1:           # (1,H,W)
            return imgs[0].float()
        if imgs.dim() == 4:
            if imgs.shape[-1] == 1:                          # (1,H,W,1)
                return imgs[0, :, :, 0].float()
            if imgs.shape[-1] == 4:                          # (1,H,W,4) -> RGBD
                return imgs[0, :, :, 3].float()

    # B) 直接从 meta_img（Tensor）取
    if allow_meta_img and isinstance(meta_img, torch.Tensor):
        if meta_img.dim() == 3 and meta_img.shape[0] == 1:    # (1,H,W)
            return meta_img[0].float()
        if meta_img.dim() == 4 and meta_img.shape[-1] == 1:   # (1,H,W,1)
            return meta_img[0, :, :, 0].float()

    # C) 从 dict 里按 key 取
    def _pick_from_dict(d: dict, keys):
        if not isinstance(d, dict):
            return None
        for k in keys:
            if k in d:
                v = d[k]
                if isinstance(v, torch.Tensor):
                    if v.dim() == 3 and v.shape[0] == 1:
                        return v[0].float()
                    if v.dim() == 2:
                        return v.float()
                    if v.dim() == 4 and v.shape[-1] == 1:
                        return v[0, :, :, 0].float()
        return None

    if prefer_key is not None:
        cand = _pick_from_dict(meta_img, [prefer_key]) or _pick_from_dict(meta, [prefer_key])
        if cand is not None:
            return cand

    # 注意：把 'depths' 放在首位，以兼容你当前分支；不过它往往是 per-gauss 1D，不一定是像素深度
    common_keys = ["depths", "depth", "depth_map", "Depth", "zbuf", "z_buffer", "z"]
    cand = _pick_from_dict(meta_img, common_keys) or _pick_from_dict(meta, common_keys)
    if cand is not None:
        return cand

    m_img_keys = list(meta_img.keys()) if isinstance(meta_img, dict) else type(meta_img).__name__
    m_keys = list(meta.keys()) if isinstance(meta, dict) else type(meta).__name__
    raise RuntimeError(
        f"未能解析深度图。imgs.shape="
        f"{tuple(imgs.shape) if isinstance(imgs, torch.Tensor) else type(imgs).__name__}; "
        f"meta_img keys={m_img_keys}; meta keys={m_keys}."
    )

def _save_depth(path: str, depth_m: np.ndarray, depth_format: str, depth_scale: float):
    """
    保存深度图：
      - uint16_png : 保存为 16-bit PNG（毫米：depth_mm = clamp(depth_m*scale, 0, 65535)）
      - float32_npy: 保存为 .npy（单位米）
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    depth_format = depth_format.lower()
    if depth_format == "uint16_png":
        depth_mm = np.clip(depth_m * depth_scale, 0, 65535).astype(np.uint16)
        imageio.imwrite(path, depth_mm)
    elif depth_format == "float32_npy":
        if not path.endswith(".npy"):
            path = path + ".npy"
        np.save(path, depth_m.astype(np.float32))
    else:
        raise ValueError(f"不支持的 depth_format: {depth_format}（可选 'uint16_png' 或 'float32_npy'）")


# ------------------ 主函数：深度渲染 ------------------

@torch.no_grad()
def render_depth_and_save_rt(
    poses_rt: Iterable[RT],
    img_paths: Iterable[str],
    model: Dict[str, ArrayLike],
    intrinsics: Dict[str, ArrayLike],
    device: str = "cuda",
    rasterize_mode: str = "classic",
    radius_clip: float = 0.0,
    depth_format: str = "uint16_png",    # "uint16_png" | "float32_npy"
    depth_scale: float = 1000.0,         # 16bit PNG 时用（米->毫米）
) -> List[str]:
    """
    按 (R,t)（world->camera）逐帧渲染深度并保存。
    返回：保存成功的路径列表（与输入 img_paths 对齐）
    """
    device = device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
    mdl = _ensure_model_tensors(model, device)
    K_torch, W, H = _make_K_tensor(intrinsics, device)

    poses_rt = list(poses_rt)
    img_paths = list(img_paths)
    if len(poses_rt) != len(img_paths):
        raise ValueError("poses_rt 和 img_paths 数量不匹配")

    saved: List[str] = []
    printed_once = False

    for idx, ((R, t), path) in enumerate(zip(poses_rt, img_paths)):
        viewmats = _rt_to_viewmat(R, t, device)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            imgs, meta_img, meta = gs.rasterization(
                means=mdl["means"], quats=mdl["quats"], scales=mdl["scales"],
                opacities=mdl["opacities"], colors=mdl["colors"],
                viewmats=viewmats, Ks=K_torch, width=W, height=H,
                render_mode="D", rasterize_mode=rasterize_mode, radius_clip=radius_clip
            )
        except Exception as err:
            raise RuntimeError(f"Depth rasterization failed at frame {idx} ({path}): {err}") from err

        if not printed_once:
            _debug_print_render_outputs(imgs, meta_img, meta, tag="[RGBD]")
            printed_once = True

        depth_tensor = _extract_depth(imgs, meta_img, meta, prefer_key=None, allow_meta_img=True)
        depth = depth_tensor.detach().cpu().numpy()
        _save_depth(path, depth, depth_format=depth_format, depth_scale=depth_scale)
        saved.append(path)
        print(f"[depth_render] Saved depth: {path}")

    return saved

@torch.no_grad()
def render_depth_and_save_Twc(
    poses_Twc: Iterable[ArrayLike],
    img_paths: Iterable[str],
    model: Dict[str, ArrayLike],
    intrinsics: Dict[str, ArrayLike],
    device: str = "cuda",
    rasterize_mode: str = "classic",
    radius_clip: float = 0.0,
    depth_format: str = "uint16_png",
    depth_scale: float = 1000.0,
) -> List[str]:
    """
    传入 T_wc 列表（camera 在世界中的位姿），内部转换为 (R,t) 后渲染深度并保存。
    """
    poses_rt: List[RT] = []
    for T in poses_Twc:
        R_wc, t_wc = _twc_to_rt(T)
        poses_rt.append((R_wc, t_wc))
    return render_depth_and_save_rt(
        poses_rt=poses_rt,
        img_paths=img_paths,
        model=model,
        intrinsics=intrinsics,
        device=device,
        rasterize_mode=rasterize_mode,
        radius_clip=radius_clip,
        depth_format=depth_format,
        depth_scale=depth_scale
    )
