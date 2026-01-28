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

"""Frame transformation utilities for dexterous manipulation primitives.

This module provides functions for:
- End-effector pose computation
- Grasp frame transformations
- Relative pose calculations
- Synergy space projections
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Tuple

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    subtract_frame_transforms,
    quat_error_magnitude,
    quat_mul,
    quat_inv,
    matrix_from_quat,
    quat_from_matrix,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "get_ee_pose_in_robot_frame",
    "get_ee_pose_in_world_frame",
    "compute_pose_error",
    "compute_ee_twist",
    "get_fingertip_positions",
    "compute_grasp_center",
    "project_to_synergy_space",
]


def get_ee_pose_in_robot_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    frame_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get end-effector pose in robot root frame.

    Args:
        env: The environment instance
        ee_frame_cfg: Scene entity config for the EE frame transformer
        robot_cfg: Scene entity config for the robot
        frame_idx: Index of the target frame in the frame transformer

    Returns:
        Tuple of (position, quaternion) in robot root frame
        - position: (num_envs, 3)
        - quaternion: (num_envs, 4) in (w, x, y, z) format
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    robot = env.scene[robot_cfg.name]

    # Get EE pose in world frame
    ee_pos_w = ee_frame.data.target_pos_w[..., frame_idx, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., frame_idx, :]

    # Get robot root pose
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    # Transform EE pose to robot frame
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w,
        ee_pos_w, ee_quat_w
    )

    return ee_pos_b, ee_quat_b


def get_ee_pose_in_world_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    frame_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get end-effector pose in world frame.

    Args:
        env: The environment instance
        ee_frame_cfg: Scene entity config for the EE frame transformer
        frame_idx: Index of the target frame in the frame transformer

    Returns:
        Tuple of (position, quaternion) in world frame
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., frame_idx, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., frame_idx, :]
    return ee_pos_w, ee_quat_w


def compute_pose_error(
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute position and orientation error between current and target poses.

    Args:
        current_pos: Current position (num_envs, 3)
        current_quat: Current quaternion (num_envs, 4)
        target_pos: Target position (num_envs, 3)
        target_quat: Target quaternion (num_envs, 4)

    Returns:
        Tuple of:
        - position_error: L2 norm of position difference (num_envs,)
        - orientation_error: Angle between quaternions in radians (num_envs,)
    """
    position_error = torch.norm(current_pos - target_pos, dim=-1)
    orientation_error = quat_error_magnitude(current_quat, target_quat)
    return position_error, orientation_error


def compute_ee_twist(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_body_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute end-effector twist (linear and angular velocity).

    Args:
        env: The environment instance
        robot_cfg: Scene entity config for the robot
        ee_body_name: Name of the end-effector body

    Returns:
        Tuple of:
        - linear_vel: Linear velocity (num_envs, 3)
        - angular_vel: Angular velocity (num_envs, 3)
    """
    robot = env.scene[robot_cfg.name]

    # Get body index
    # TODO: Cache this lookup for efficiency
    body_idx = robot.find_bodies(ee_body_name)[0][0]

    linear_vel = robot.data.body_lin_vel_w[:, body_idx, :]
    angular_vel = robot.data.body_ang_vel_w[:, body_idx, :]

    return linear_vel, angular_vel


def get_fingertip_positions(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    fingertip_body_names: list,
) -> torch.Tensor:
    """Get positions of all fingertips.

    Args:
        env: The environment instance
        robot_cfg: Scene entity config for the robot
        fingertip_body_names: List of fingertip body names

    Returns:
        Tensor of shape (num_envs, num_fingers, 3) with fingertip positions
    """
    robot = env.scene[robot_cfg.name]

    # Get body indices
    body_indices = []
    for name in fingertip_body_names:
        idx = robot.find_bodies(name)[0][0]
        body_indices.append(idx)

    # Extract positions
    positions = robot.data.body_pos_w[:, body_indices, :]
    return positions


def compute_grasp_center(
    fingertip_positions: torch.Tensor,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """Compute grasp center from fingertip positions.

    Args:
        fingertip_positions: (num_envs, num_fingers, 3)
        weights: Optional weights per finger (num_fingers,)

    Returns:
        Grasp center position (num_envs, 3)
    """
    if weights is None:
        return fingertip_positions.mean(dim=1)
    else:
        weights = weights / weights.sum()
        weighted_sum = (fingertip_positions * weights.view(1, -1, 1)).sum(dim=1)
        return weighted_sum


def project_to_synergy_space(
    joint_positions: torch.Tensor,
    synergy_matrix: torch.Tensor,
) -> torch.Tensor:
    """Project joint positions to synergy space.

    Synergies are low-dimensional representations of hand configurations
    that capture common grasp patterns.

    Args:
        joint_positions: Hand joint positions (num_envs, num_joints)
        synergy_matrix: Synergy basis matrix (num_synergies, num_joints)
                       Each row is a synergy vector

    Returns:
        Synergy coefficients (num_envs, num_synergies)
    """
    # Project: coeffs = joint_positions @ synergy_matrix.T
    # Using pseudoinverse for robustness
    synergy_coeffs = torch.matmul(joint_positions, synergy_matrix.T)
    return synergy_coeffs


def reconstruct_from_synergies(
    synergy_coeffs: torch.Tensor,
    synergy_matrix: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct joint positions from synergy coefficients.

    Args:
        synergy_coeffs: Synergy coefficients (num_envs, num_synergies)
        synergy_matrix: Synergy basis matrix (num_synergies, num_joints)

    Returns:
        Reconstructed joint positions (num_envs, num_joints)
    """
    return torch.matmul(synergy_coeffs, synergy_matrix)


def compute_synergy_error(
    current_joint_pos: torch.Tensor,
    target_synergy_coeffs: torch.Tensor,
    synergy_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute error between current joints and target synergy configuration.

    Args:
        current_joint_pos: Current joint positions (num_envs, num_joints)
        target_synergy_coeffs: Target synergy coefficients (num_envs, num_synergies)
        synergy_matrix: Synergy basis matrix (num_synergies, num_joints)

    Returns:
        Synergy tracking error (num_envs,)
    """
    # Project current joints to synergy space
    current_coeffs = project_to_synergy_space(current_joint_pos, synergy_matrix)

    # Compute error in synergy space
    synergy_error = torch.norm(current_coeffs - target_synergy_coeffs, dim=-1)

    return synergy_error
