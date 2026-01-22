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
from isaaclab.utils.math import quat_apply, quat_mul, quat_from_euler_xyz, sample_uniform


def reset_bimanual_objects_symmetric(
    env,
    env_ids: Sequence[int],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    left_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_body_name: str | None = None,
) -> None:
    """Reset left/right objects symmetrically in the robot root frame."""
    left_obj: RigidObject = env.scene[left_cfg.name]
    right_obj: RigidObject = env.scene[right_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    left_states = left_obj.data.default_root_state[env_ids].clone()
    right_states = right_obj.data.default_root_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=left_obj.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=left_obj.device)

    pos_offset_left = rand_samples[:, 0:3]
    pos_offset_right = pos_offset_left.clone()
    pos_offset_right[:, 1] *= -1.0

    if base_body_name is not None and base_body_name in robot.data.body_names:
        body_idx = robot.data.body_names.index(base_body_name)
        robot_pos_w = robot.data.body_pos_w[env_ids, body_idx]
        robot_quat_w = robot.data.body_quat_w[env_ids, body_idx]
    else:
        robot_pos_w = robot.data.root_pos_w[env_ids]
        robot_quat_w = robot.data.root_quat_w[env_ids]

    left_pos_w = robot_pos_w + quat_apply(robot_quat_w, pos_offset_left)
    right_pos_w = robot_pos_w + quat_apply(robot_quat_w, pos_offset_right)

    orient_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    left_quat_w = quat_mul(left_states[:, 3:7], orient_delta)
    right_quat_w = quat_mul(right_states[:, 3:7], orient_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=left_obj.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=left_obj.device)
    left_vel = left_states[:, 7:13] + rand_samples
    right_vel = right_states[:, 7:13] + rand_samples

    left_obj.write_root_pose_to_sim(torch.cat([left_pos_w, left_quat_w], dim=-1), env_ids=env_ids)
    right_obj.write_root_pose_to_sim(torch.cat([right_pos_w, right_quat_w], dim=-1), env_ids=env_ids)
    left_obj.write_root_velocity_to_sim(left_vel, env_ids=env_ids)
    right_obj.write_root_velocity_to_sim(right_vel, env_ids=env_ids)


def reset_bimanual_objects_symmetric_world(
    env,
    env_ids: Sequence[int],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    left_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("object2"),
) -> None:
    """Reset left/right objects symmetrically in world frame (mirror across Y)."""
    left_obj: RigidObject = env.scene[left_cfg.name]
    right_obj: RigidObject = env.scene[right_cfg.name]

    left_states = left_obj.data.default_root_state[env_ids].clone()
    right_states = right_obj.data.default_root_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=left_obj.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=left_obj.device)

    pos_left = rand_samples[:, 0:3]
    pos_right = pos_left.clone()
    pos_right[:, 1] *= -1.0

    orient_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    left_quat = quat_mul(left_states[:, 3:7], orient_delta)
    right_quat = quat_mul(right_states[:, 3:7], orient_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=left_obj.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=left_obj.device)
    left_vel = left_states[:, 7:13] + rand_samples
    right_vel = right_states[:, 7:13] + rand_samples

    left_obj.write_root_pose_to_sim(torch.cat([pos_left, left_quat], dim=-1), env_ids=env_ids)
    right_obj.write_root_pose_to_sim(torch.cat([pos_right, right_quat], dim=-1), env_ids=env_ids)
    left_obj.write_root_velocity_to_sim(left_vel, env_ids=env_ids)
    right_obj.write_root_velocity_to_sim(right_vel, env_ids=env_ids)


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
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    pos_offset = rand_samples[:, 0:3]
    if base_body_name is not None and base_body_name in robot.data.body_names:
        body_idx = robot.data.body_names.index(base_body_name)
        robot_pos_w = robot.data.body_pos_w[env_ids, body_idx]
        robot_quat_w = robot.data.body_quat_w[env_ids, body_idx]
    else:
        robot_pos_w = robot.data.root_pos_w[env_ids]
        robot_quat_w = robot.data.root_quat_w[env_ids]
    positions = robot_pos_w + quat_apply(robot_quat_w, pos_offset)

    orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + rand_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
