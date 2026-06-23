# SPDX-License-Identifier: Apache-2.0
# act2cam control for LingBot-World-Act, adapted from
# https://github.com/Robbyant/lingbot-world (wan/utils/wasd_ijkl_to_c2ws.py).
"""Build the 7-channel act2cam control tensor (rays_d + WASD) for LingBot-World-Act.

Discrete WASD (move) / IJKL (look) actions -> a camera trajectory -> Plucker
ray-direction embeddings (3 ch) concatenated with the per-frame WASD channels (4),
giving control_dim=7 as expected by the Act DiT (patch_embedding_wancamctrl 7*64).
"""
import numpy as np
import torch

from fastvideo.models.dits.lingbotworld.cam_utils import (compute_relative_poses,
                                                          create_meshgrid,
                                                          interpolate_camera_poses)

_ALLOWED_KEYS = frozenset("wasdijkl")
_WASD_IDX = {"w": 0, "a": 1, "s": 2, "d": 3}
_IJKL_IDX = {"i": 0, "j": 1, "k": 2, "l": 3}


def parse_action_string(action_string: str):
    s = action_string.replace("，", ",")
    s = "".join(s.split())
    segments = []
    for part in [p for p in s.split(",") if p]:
        keys_part, dur = part.rsplit("-", 1)
        n = int(dur)
        keys = frozenset() if keys_part.lower() == "none" else frozenset(keys_part.lower())
        assert not (keys - _ALLOWED_KEYS), f"bad keys in {part!r}"
        segments.append((keys, n))
    return segments


def action_string_to_wasd_ijkl(action_string: str):
    segments = parse_action_string(action_string)
    total = sum(n for _, n in segments)
    wasd = np.zeros((total, 4), dtype=np.float32)
    ijkl = np.zeros((total, 4), dtype=np.float32)
    t = 0
    for keys, n in segments:
        for _ in range(n):
            for c in keys:
                (wasd if c in _WASD_IDX else ijkl)[t, (_WASD_IDX | _IJKL_IDX)[c]] = 1.0
            t += 1
    return wasd, ijkl, total


def _rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])}[axis]


def generate_trajectory(wasd: np.ndarray, ijkl: np.ndarray) -> np.ndarray:
    """Per-frame WASD/IJKL key states -> list of 4x4 camera-to-world matrices."""
    move_speed, rot_speed = 0.05, np.deg2rad(2.0)
    pitch_limit = np.deg2rad(85)
    c2w = np.eye(4)
    pitch = 0.0
    out = [c2w.copy()]
    for f in range(len(wasd)):
        w, a, s, d = wasd[f] > 0.5
        i, j, k, l = ijkl[f] > 0.5
        R = c2w[:3, :3]
        pitch_delta = (rot_speed if i else 0.0) - (rot_speed if k else 0.0)
        if not (-pitch_limit <= pitch + pitch_delta <= pitch_limit):
            pitch_delta = 0.0
        pitch += pitch_delta
        yaw_delta = (rot_speed if l else 0.0) - (rot_speed if j else 0.0)
        R_new = _rot("y", yaw_delta) @ R @ _rot("x", pitch_delta)
        fwd = np.array([R_new[0, 2], 0, R_new[2, 2]])
        rgt = np.array([R_new[0, 0], 0, R_new[2, 0]])
        fwd = fwd / (np.linalg.norm(fwd) + 1e-6)
        rgt = rgt / (np.linalg.norm(rgt) + 1e-6)
        move = (fwd * move_speed * (int(w) - int(s)) + rgt * move_speed * (int(d) - int(a)))
        c2w = np.eye(4)
        c2w[:3, :3] = R_new
        c2w[:3, 3] = c2w[:3, 3] * 0 + (out[-1][:3, 3] + move)
        out.append(c2w.copy())
    return np.array(out)


def _plucker_rays_d(c2ws_mat: torch.Tensor, Ks: torch.Tensor, height: int, width: int) -> torch.Tensor:
    n = c2ws_mat.shape[0]
    grid = create_meshgrid(n, height, width, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    fx, fy, cx, cy = Ks.chunk(4, dim=-1)
    i, j = grid[..., 0], grid[..., 1]
    z = torch.ones_like(i)
    dirs = torch.stack([(i - cx) / fx * z, (j - cy) / fy * z, z], dim=-1)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    rays_d = dirs @ c2ws_mat[:, :3, :3].transpose(-1, -2)
    return rays_d.view(n, height, width, 3)


def _fold(x: torch.Tensor, lat_h: int, lat_w: int, c: int, s: int) -> torch.Tensor:
    # [F, lat_h*s, lat_w*s, c] -> [F, lat_h, lat_w, c*s*s]
    f = x.shape[0]
    x = x.view(f, lat_h, s, lat_w, s, c).permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(f, lat_h, lat_w, c * s * s)


def prepare_action_embedding(
    action_string: str,
    num_frames: int,
    height: int,
    width: int,
    spatial_scale: int = 8,
    focal: float | None = None,
) -> tuple[torch.Tensor, int]:
    """action_string (e.g. 'w-20,wl-12,none-8,s-8') -> (c2ws_plucker_emb, num_frames).

    Returns control tensor [1, 7*spatial_scale^2, lat_f, lat_h, lat_w] and the
    (4n+1-aligned) frame count.
    """
    wasd, ijkl, total = action_string_to_wasd_ijkl(action_string)
    num_frames = min(num_frames, ((total - 1) // 4) * 4 + 1)
    num_frames = ((num_frames - 1) // 4) * 4 + 1
    wasd, ijkl = wasd[:num_frames], ijkl[:num_frames]

    c2ws = generate_trajectory(wasd, ijkl)[:num_frames]
    num_lat = (num_frames - 1) // 4 + 1
    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, num_frames - 1, num_frames),
        src_rot_mat=c2ws[:, :3, :3],
        src_trans_vec=c2ws[:, :3, 3],
        tgt_indices=np.linspace(0, num_frames - 1, num_lat),
    )
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)

    focal = focal if focal is not None else float(width)
    Ks = torch.tensor([[focal, focal, width / 2.0, height / 2.0]], dtype=torch.float32).repeat(num_lat, 1)

    lat_h, lat_w = height // spatial_scale, width // spatial_scale
    rays_d = _plucker_rays_d(c2ws_infer, Ks, height, width)           # [F, H, W, 3]
    rays_d = _fold(rays_d, lat_h, lat_w, 3, spatial_scale)            # [F, lat_h, lat_w, 3*64]

    wasd_lat = torch.from_numpy(wasd[::4][:num_lat]).float()          # [F, 4]
    wasd_grid = wasd_lat[:, None, None, :].repeat(1, height, width, 1)  # [F, H, W, 4]
    wasd_grid = _fold(wasd_grid, lat_h, lat_w, 4, spatial_scale)      # [F, lat_h, lat_w, 4*64]

    ctrl = torch.cat([rays_d, wasd_grid], dim=-1)                    # [F, lat_h, lat_w, 7*64]
    c2ws_plucker_emb = ctrl.permute(3, 0, 1, 2).contiguous().unsqueeze(0)  # [1, 7*64, F, lat_h, lat_w]
    return c2ws_plucker_emb, num_frames
