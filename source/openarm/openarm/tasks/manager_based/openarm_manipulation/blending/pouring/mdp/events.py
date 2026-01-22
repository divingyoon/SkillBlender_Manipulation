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

from typing import Sequence

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply
import isaaclab.utils.math as math_utils


def reset_bead_in_cup(
    env,
    env_ids: Sequence[int],
    cup_name: str = "object",
    bead_name: str = "bead",
    offset: tuple[float, float, float] = (0.0, 0.0, 0.05),
) -> None:
    """Reset bead pose to be inside the source cup."""
    cup: RigidObject = env.scene[cup_name]
    bead: RigidObject = env.scene[bead_name]

    cup_pos = cup.data.root_pos_w[env_ids]
    cup_quat = cup.data.root_quat_w[env_ids]

    offset_vec = torch.tensor(offset, device=env.device, dtype=torch.float32)
    offset_vec = offset_vec.expand(cup_pos.shape[0], 3)
    bead_pos = cup_pos + quat_apply(cup_quat, offset_vec)
    bead_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(cup_pos.shape[0], 4)

    zeros = torch.zeros_like(bead_pos)
    root_state = torch.cat([bead_pos, bead_quat, zeros, zeros], dim=-1)
    bead.write_root_state_to_sim(root_state, env_ids=env_ids)


def reset_root_state_uniform_robot_frame(
    env,
    env_ids: Sequence[int],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_body_name: str | None = None,
) -> None:
    """Reset asset root using a pose sampled in the robot root frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    root_states = asset.data.default_root_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    pos_offset = rand_samples[:, 0:3]
    if base_body_name is not None and base_body_name in robot.data.body_names:
        body_idx = robot.data.body_names.index(base_body_name)
        robot_pos_w = robot.data.body_pos_w[env_ids, body_idx]
        robot_quat_w = robot.data.body_quat_w[env_ids, body_idx]
    else:
        robot_pos_w = robot.data.root_pos_w[env_ids]
        robot_quat_w = robot.data.root_quat_w[env_ids]
    positions = robot_pos_w + quat_apply(robot_quat_w, pos_offset)

    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + rand_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
