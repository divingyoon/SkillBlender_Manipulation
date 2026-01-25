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

"""
Event functions for the Grasp-v1 task.

Key concept: Teacher-forcing reset
- Cup is placed on table with some randomization (xy offset, yaw rotation)
- Robot arm is initialized to pre-grasp pose relative to cup
- Gripper starts fully open
"""

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    quat_apply,
    quat_mul,
    quat_from_euler_xyz,
    sample_uniform,
    quat_inv,
)


def reset_cup_with_randomization(
    env,
    env_ids: Sequence[int],
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> None:
    """Reset cup on table with position and yaw randomization.

    Args:
        pose_range: Dictionary with keys 'x', 'y', 'z', 'yaw' for randomization ranges.
                   Example: {'x': (-0.03, 0.03), 'y': (-0.03, 0.03), 'yaw': (-0.26, 0.26)}
    """
    cup: RigidObject = env.scene[asset_cfg.name]

    default_state = cup.data.default_root_state[env_ids].clone()

    # Sample random offsets
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=cup.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=cup.device)

    # Apply position offset (add to default position + env origin)
    positions = default_state[:, :3] + env.scene.env_origins[env_ids]
    positions[:, 0] += rand_samples[:, 0]  # x offset
    positions[:, 1] += rand_samples[:, 1]  # y offset
    positions[:, 2] += rand_samples[:, 2]  # z offset

    # Apply orientation delta (yaw only for cup)
    orient_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = quat_mul(default_state[:, 3:7], orient_delta)

    # Zero velocity
    velocities = torch.zeros((len(env_ids), 6), device=cup.device)

    cup.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    cup.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_robot_to_pregrasp_pose(
    env,
    env_ids: Sequence[int],
    pregrasp_distance: tuple[float, float],
    height_offset: tuple[float, float],
    yaw_noise: tuple[float, float],
    target_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    tcp_body_name: str = "openarm_left_ee_tcp",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot arm to pre-grasp pose relative to cup.

    This uses IK or direct joint positioning to place the TCP at a pre-grasp location:
    - Distance from cup: pregrasp_distance (e.g., 3-6cm in front)
    - Height: cup center height + height_offset
    - Yaw alignment: roughly aligned with some noise

    For now, we use a simplified approach: reset to default joint positions
    which should already be close to a pre-grasp pose.
    The actual pre-grasp positioning is handled by the reward structure.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    # Reset to default joint positions (pre-grasp pose defined in config)
    default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = robot.data.default_joint_vel[env_ids].clone()

    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)


def reset_gripper_open(
    env,
    env_ids: Sequence[int],
    joint_names: list[str],
    open_position: float = 0.04,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset gripper joints to fully open position.

    Args:
        joint_names: List of gripper joint names to reset.
        open_position: Target position for open gripper (e.g., 0.04 for fully open).
    """
    robot: Articulation = env.scene[robot_cfg.name]

    joint_pos = robot.data.joint_pos[env_ids].clone()
    joint_vel = robot.data.joint_vel[env_ids].clone()

    for joint_name in joint_names:
        if joint_name in robot.data.joint_names:
            joint_idx = robot.data.joint_names.index(joint_name)
            joint_pos[:, joint_idx] = open_position
            joint_vel[:, joint_idx] = 0.0

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_robot_to_pregrasp_ik(
    env,
    env_ids: Sequence[int],
    pregrasp_offset: list[float],
    offset_noise: dict[str, tuple[float, float]],
    target_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    tcp_body_name: str = "openarm_left_ee_tcp",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot TCP to pre-grasp pose using offset from cup position.

    This computes the desired TCP pose as:
        TCP_target = cup_pos + rotate(cup_quat, pregrasp_offset + noise)

    Then sets joint positions to achieve this pose.
    Note: This is a placeholder - actual IK solver integration needed.

    Args:
        pregrasp_offset: Base offset from cup center in cup's local frame [x, y, z].
                        e.g., [0.0, 0.0, -0.05] means 5cm behind the cup.
        offset_noise: Random noise to add to offset {'x': (min, max), 'y': ..., 'z': ...}
    """
    robot: Articulation = env.scene[robot_cfg.name]
    cup: RigidObject = env.scene[target_cfg.name]

    # Get cup pose
    cup_pos = cup.data.root_pos_w[env_ids]
    cup_quat = cup.data.root_quat_w[env_ids]

    # Compute target TCP position
    base_offset = torch.tensor(pregrasp_offset, device=env.device).unsqueeze(0)
    base_offset = base_offset.expand(len(env_ids), -1).clone()

    # Add noise
    range_list = [offset_noise.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)
    noise = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=env.device)
    offset_with_noise = base_offset + noise

    # Transform offset to world frame
    offset_world = quat_apply(cup_quat, offset_with_noise)
    target_tcp_pos = cup_pos + offset_world

    # For now, just reset to default - IK integration needed for actual positioning
    # Store target for observation/reward computation
    if not hasattr(env, "_pregrasp_target_pos"):
        env._pregrasp_target_pos = torch.zeros((env.num_envs, 3), device=env.device)
    env._pregrasp_target_pos[env_ids] = target_tcp_pos

    # Reset to default joint positions
    default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = robot.data.default_joint_vel[env_ids].clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
