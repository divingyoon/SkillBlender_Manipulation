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

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from isaaclab.envs import ManagerBasedRLEnv


def bead_spill(
    env: ManagerBasedRLEnv,
    bead_name: str,
    target_name: str,
    min_height_offset: float = -0.02,
    xy_radius: float = 0.08,
) -> torch.Tensor:
    """Terminate when the bead drops below the target cup height outside the cup area."""
    bead_pos = env.scene[bead_name].data.root_pos_w - env.scene.env_origins
    target_pos = env.scene[target_name].data.root_pos_w - env.scene.env_origins
    spill_height = target_pos[:, 2] + min_height_offset
    d_xy = torch.norm(bead_pos[:, :2] - target_pos[:, :2], p=2, dim=-1)
    return (bead_pos[:, 2] < spill_height) & (d_xy > xy_radius)


def cup_tipped(
    env: ManagerBasedRLEnv,
    object_name: str,
    min_upright_dot: float = 0.5,
) -> torch.Tensor:
    """Terminate when the cup tilts beyond the upright threshold."""
    object_quat_w = env.scene[object_name].data.root_quat_w
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=object_quat_w.dtype)
    cup_z_axis = quat_apply(object_quat_w, z_axis.expand(object_quat_w.shape[0], 3))
    dot = torch.sum(cup_z_axis * z_axis, dim=1)
    return dot < min_upright_dot


def object_out_of_reach(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_xy: float = 0.7,
    max_z: float = 0.4,
    min_z: float = -0.05,
) -> torch.Tensor:
    """Terminate when the object is outside a reachable box in robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w[:, :3]
    )
    xy_norm = torch.norm(object_pos_b[:, :2], p=2, dim=-1)
    z = object_pos_b[:, 2]
    return (xy_norm > max_xy) | (z > max_z) | (z < min_z)
