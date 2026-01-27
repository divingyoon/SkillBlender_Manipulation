# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch

from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_angle_axis, quat_from_matrix, quat_mul, quat_unique


class YAxisAlignedPoseCommand(UniformPoseCommand):
    """Pose command generator with body +Y aligned to world +X."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)

        num_ids = len(env_ids)
        device = self.device
        dtype = self.pose_command_b.dtype

        y_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).repeat(num_ids, 1)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).repeat(num_ids, 1)
        x_axis = torch.cross(y_axis, z_axis, dim=-1)
        rot = torch.stack((x_axis, y_axis, z_axis), dim=-1)
        quat = quat_from_matrix(rot)

        spin = torch.empty((num_ids,), device=device, dtype=dtype)
        spin.uniform_(*self.cfg.ranges.roll)
        spin_quat = quat_from_angle_axis(spin, y_axis)
        quat = quat_mul(spin_quat, quat)

        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat


@configclass
class YAxisAlignedPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for YAxisAlignedPoseCommand."""

    class_type: type = YAxisAlignedPoseCommand
