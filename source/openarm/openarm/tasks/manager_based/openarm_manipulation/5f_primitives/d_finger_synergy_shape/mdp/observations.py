# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0

"""Observation functions for Primitive D: Finger Joint Target Tracking."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


__all__ = ["joint_limit_margins"]


def joint_limit_margins(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Get normalized margin from joint limits.

    Returns 1.0 at center of range, 0.0 at limits.

    Args:
        env: Environment instance
        asset_cfg: Asset config with joint_names

    Returns:
        Joint margins (num_envs, num_joints)
    """
    asset = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos
    if asset_cfg.joint_ids is not None:
        joint_pos = joint_pos[:, asset_cfg.joint_ids]

    joint_limits = asset.data.soft_joint_pos_limits
    if asset_cfg.joint_ids is not None:
        joint_limits = joint_limits[:, asset_cfg.joint_ids, :]

    lower = joint_limits[..., 0]
    upper = joint_limits[..., 1]
    range_size = upper - lower

    # Normalize to [0, 1]
    normalized = (joint_pos - lower) / (range_size + 1e-8)

    # Parabolic margin: peaks at 0.5
    margin = 4.0 * normalized * (1.0 - normalized)

    return torch.clamp(margin, 0.0, 1.0)
