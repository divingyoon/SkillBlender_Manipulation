"""
Termination terms for GraspIK.
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
