"""
Termination terms for TransferIK.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply
from isaaclab.managers import SceneEntityCfg


def joint_limit_near_or_min_margin(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin_threshold: float = 0.01,
    near_rate_threshold: float = 0.9,
    warmup_steps: int = 0,
) -> torch.Tensor:
    """Terminate when joints are too close to limits.

    Criteria (per-env):
    - min margin to limit < margin_threshold, OR
    - fraction of joints within margin_threshold > near_rate_threshold.
    """
    if warmup_steps > 0:
        in_warmup = env.episode_length_buf < warmup_steps
    else:
        in_warmup = None
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    limits = asset.data.soft_joint_pos_limits[:, joint_ids, :]
    q = asset.data.joint_pos[:, joint_ids]

    margin = torch.minimum(q - limits[..., 0], limits[..., 1] - q)
    min_margin = margin.min(dim=1).values
    near_rate = (margin < margin_threshold).float().mean(dim=1)

    triggered = (min_margin < margin_threshold) | (near_rate > near_rate_threshold)
    if in_warmup is not None:
        triggered = triggered & (~in_warmup)
    return triggered


def cup_tipped(
    env,
    object_name: str,
    min_upright_dot: float = 0.5,
) -> torch.Tensor:
    """Terminate when the cup tilts beyond the upright threshold."""
    object_quat_w = env.scene[object_name].data.root_quat_w
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=object_quat_w.dtype)
    cup_z_axis = quat_apply(object_quat_w, z_axis.expand(object_quat_w.shape[0], 3))
    dot = torch.sum(cup_z_axis * z_axis, dim=1)
    return dot < min_upright_dot


def bead_spill(
    env,
    bead_name: str,
    target_name: str,
    min_height_offset: float = -0.04,
    xy_radius: float = 0.08,
) -> torch.Tensor:
    """Terminate when the bead drops below the target cup height outside the cup area."""
    bead_pos = env.scene[bead_name].data.root_pos_w - env.scene.env_origins
    target_pos = env.scene[target_name].data.root_pos_w - env.scene.env_origins
    spill_height = target_pos[:, 2] + min_height_offset
    d_xy = torch.norm(bead_pos[:, :2] - target_pos[:, :2], p=2, dim=-1)
    return (bead_pos[:, 2] < spill_height) & (d_xy > xy_radius)


def object2_close_to_object(
    env,
    target_name: str = "object",
    source_name: str = "object2",
    threshold: float = 0.05,
    target_offset: tuple[float, float, float] = (0.0, 0.0, 0.10),
    source_offset: tuple[float, float, float] = (0.0, 0.0, 0.10),
) -> torch.Tensor:
    """Terminate when source cup is close to target cup (with offsets)."""
    target_pos = env.scene[target_name].data.root_pos_w
    source_pos = env.scene[source_name].data.root_pos_w
    device = target_pos.device
    dtype = target_pos.dtype
    target_off = torch.tensor(target_offset, device=device, dtype=dtype)
    source_off = torch.tensor(source_offset, device=device, dtype=dtype)
    target_pos = target_pos + target_off
    source_pos = source_pos + source_off
    dist = torch.norm(source_pos - target_pos, dim=1)
    return dist < threshold
