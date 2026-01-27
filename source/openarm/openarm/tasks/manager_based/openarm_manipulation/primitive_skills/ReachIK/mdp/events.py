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
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms
import os
import isaaclab.utils.math as math_utils

_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../../../../")
)
_OBS_LOG_PATH = os.path.join(_ROOT_DIR, "obs_debug.log")


def _append_obs_log(line: str) -> None:
    try:
        with open(_OBS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


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

    # Position randomization (x, y, z) - all randomized simultaneously
    range_list_pos = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges_pos = torch.tensor(range_list_pos, device=asset.device)
    pos_offset = math_utils.sample_uniform(ranges_pos[:, 0], ranges_pos[:, 1], (len(env_ids), 3), device=asset.device)

    if base_body_name is not None and base_body_name in robot.data.body_names:
        body_idx = robot.data.body_names.index(base_body_name)
        robot_pos_w = robot.data.body_pos_w[env_ids, body_idx]
        robot_quat_w = robot.data.body_quat_w[env_ids, body_idx]
    else:
        robot_pos_w = robot.data.root_pos_w[env_ids]
        robot_quat_w = robot.data.root_quat_w[env_ids]
    positions = robot_pos_w + quat_apply(robot_quat_w, pos_offset)

    # Orientation randomization (World X, Y, or Z axis) - randomize only one at a time
    # Initialize quaternion to identity
    orientations_delta_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=asset.device).repeat(len(env_ids), 1)

    # Get ranges for roll, pitch, yaw
    roll_range = pose_range.get("roll", (0.0, 0.0))
    pitch_range = pose_range.get("pitch", (0.0, 0.0))
    yaw_range = pose_range.get("yaw", (0.0, 0.0))

    # Randomly choose one world-fixed axis (0: X, 1: Y, 2: Z) for each environment
    chosen_world_axis_idx = torch.randint(0, 3, (len(env_ids),), device=asset.device)

    # For each environment, sample an angle for the chosen world axis and create rotation quaternion
    for i in range(len(env_ids)):
        axis_idx = chosen_world_axis_idx[i].item()
        angle = torch.tensor(0.0, device=asset.device) # Default to no rotation

        if axis_idx == 0: # World X-axis
            if roll_range[1] - roll_range[0] > 1e-6:
                angle = math_utils.sample_uniform(roll_range[0], roll_range[1], (1,), device=asset.device)
            orientations_delta_quat[i] = math_utils.quat_from_angle_axis(angle, torch.tensor([1.0, 0.0, 0.0], device=asset.device))
        elif axis_idx == 1: # World Y-axis
            if pitch_range[1] - pitch_range[0] > 1e-6:
                angle = math_utils.sample_uniform(pitch_range[0], pitch_range[1], (1,), device=asset.device)
            orientations_delta_quat[i] = math_utils.quat_from_angle_axis(angle, torch.tensor([0.0, 1.0, 0.0], device=asset.device))
        elif axis_idx == 2: # World Z-axis
            if yaw_range[1] - yaw_range[0] > 1e-6:
                angle = math_utils.sample_uniform(yaw_range[0], yaw_range[1], (1,), device=asset.device)
            orientations_delta_quat[i] = math_utils.quat_from_angle_axis(angle, torch.tensor([0.0, 0.0, 1.0], device=asset.device))
    
    # Pre-multiply to apply rotation in world frame
    orientations = math_utils.quat_mul(orientations_delta_quat, root_states[:, 3:7])

    range_list_vel = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges_vel = torch.tensor(range_list_vel, device=asset.device)
    velocities = root_states[:, 7:13] + math_utils.sample_uniform(ranges_vel[:, 0], ranges_vel[:, 1], (len(env_ids), 6), device=asset.device)

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_from_command(
    env,
    env_ids: Sequence[int],
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset asset root pose to the current command pose (in robot base frame)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    command = env.command_manager.get_command(command_name)
    cmd_pos_b = command[env_ids, :3]
    cmd_quat_b = command[env_ids, 3:7]

    cmd_pos_w, cmd_quat_w = combine_frame_transforms(
        robot.data.root_pos_w[env_ids],
        robot.data.root_quat_w[env_ids],
        cmd_pos_b,
        cmd_quat_b,
    )

    zeros = torch.zeros((len(env_ids), 6), device=env.device, dtype=cmd_pos_w.dtype)
    root_state = torch.cat([cmd_pos_w, cmd_quat_w, zeros], dim=-1)
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
