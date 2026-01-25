"""
Modified command generator for the Reach‑v1 task.

This implementation adjusts the target position computation to rotate the
pre‑grasp offset from the cup's local frame into the world frame before
translation.  This ensures the commanded TCP pose stays consistent relative
to the cup even when the cup is rotated.  The orientation logic (aligning
the TCP's x-axis with the cup's z-axis) is preserved from the original
implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import math
import torch

from isaaclab.managers import CommandTerm
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_mul,
    subtract_frame_transforms,
    quat_apply,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_cfg import ReachPoseCommandCfg


class ReachPoseCommand(CommandTerm):
    """Command generator that creates a target pose relative to an object."""

    cfg: ReachPoseCommandCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: ReachPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.target_object: RigidObject = env.scene[cfg.target_asset_cfg.name]

        # command in base frame: (x, y, z, qw, qx, qy, qz)
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0

        # command in world frame (for sampling)
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        self.pose_command_w[:, 3] = 1.0

        self.offset = torch.tensor(self.cfg.offset, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # The target pose is based on the object's pose, so we just update it.
        self._update_command(env_ids)

    def _update_command(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)

        # Get cup pose
        cup_pos_w = self.target_object.data.root_pos_w[env_ids]
        cup_quat_w = self.target_object.data.root_quat_w[env_ids]

        # Compute the offset in world coordinates by rotating the local offset
        # vector using the cup's orientation.  This ensures the target position
        # remains consistent relative to the cup even when the cup rotates.
        # Expand offset to match batch for quat_apply
        offset_local = self.offset.unsqueeze(0).repeat(cup_quat_w.shape[0], 1)
        offset_world = quat_apply(cup_quat_w, offset_local)
        target_pos_w = cup_pos_w + offset_world

        # Target orientation: TCP's x-axis should align with the cup's z-axis (world up)
        # This means we want the tcp's x-axis to be (0,0,1) in world frame.
        # The original tcp orientation has x-axis pointing forward.
        # We need to rotate it by -90 degrees around the y-axis.
        batch_size = cup_pos_w.shape[0]
        rot_quat = quat_from_euler_xyz(
            torch.zeros(batch_size, device=self.device),
            -torch.ones(batch_size, device=self.device) * math.pi / 2,
            torch.zeros(batch_size, device=self.device),
        )

        # Also consider the cup's orientation
        target_quat_w = quat_mul(cup_quat_w, rot_quat)

        self.pose_command_w[env_ids, :3] = target_pos_w
        self.pose_command_w[env_ids, 3:] = target_quat_w

        # convert world command to base frame
        cmd_pos_b, cmd_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            self.pose_command_w[env_ids, :3],
            self.pose_command_w[env_ids, 3:],
        )
        self.pose_command_b[env_ids, :3] = cmd_pos_b
        self.pose_command_b[env_ids, 3:] = cmd_quat_b

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass
