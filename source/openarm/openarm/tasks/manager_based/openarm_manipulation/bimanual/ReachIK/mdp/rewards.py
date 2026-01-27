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
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_error_magnitude, quat_mul
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _hand_closure_amount(env: ManagerBasedRLEnv, eef_link_name: str) -> torch.Tensor:
    """Compute normalized closure amount for the hand associated with the given link."""
    if "left" in eef_link_name:
        hand_term = env.action_manager.get_term("left_hand_action")
    elif "right" in eef_link_name:
        hand_term = env.action_manager.get_term("right_hand_action")
    else:
        return torch.zeros(env.num_envs, device=env.device)

    hand_action = hand_term.processed_actions
    if isinstance(hand_term._offset, torch.Tensor):
        default_pos = hand_term._offset.mean(dim=1)
    else:
        default_pos = torch.full((env.num_envs,), float(hand_term._offset), device=env.device)

    mean_action = hand_action.mean(dim=1)
    return torch.clamp(
        (default_pos - mean_action) / (torch.abs(default_pos) + 1e-6), min=0.0, max=1.0
    )


def gripper_open_reward(env: ManagerBasedRLEnv, eef_link_name: str) -> torch.Tensor:
    """Reward keeping the gripper open (low closure amount)."""
    closure_amount = _hand_closure_amount(env, eef_link_name)
    return 1.0 - closure_amount

def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def position_command_within_tolerance(
    env: ManagerBasedRLEnv, tolerance: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward when position error is within a tolerance (in meters)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return (distance <= tolerance).to(distance.dtype)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking orientation using the tanh kernel.

    The function computes the shortest-path orientation error and maps it with a tanh kernel.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    error = quat_error_magnitude(curr_quat_w, des_quat_w)
    return 1 - torch.tanh(error / std)


def orientation_z_axis_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize misalignment of the body z-axis with the desired z-axis."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=curr_quat_w.device, dtype=curr_quat_w.dtype)
    z_axis = z_axis.repeat(curr_quat_w.shape[0], 1)
    curr_z = quat_apply(curr_quat_w, z_axis)
    des_z = quat_apply(des_quat_w, z_axis)
    cos_sim = torch.sum(curr_z * des_z, dim=1)
    return 1.0 - cos_sim


def hand_x_align_object_z_reward(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward aligning hand +X axis with command/object +Z axis.

    Returns a [0, 1] reward using (1 + cos(theta)) / 2.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=curr_quat_w.device, dtype=curr_quat_w.dtype)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=curr_quat_w.device, dtype=curr_quat_w.dtype)
    x_axis = x_axis.repeat(curr_quat_w.shape[0], 1)
    z_axis = z_axis.repeat(curr_quat_w.shape[0], 1)

    hand_x = quat_apply(curr_quat_w, x_axis)
    obj_z = quat_apply(des_quat_w, z_axis)
    cos_sim = torch.sum(hand_x * obj_z, dim=1)
    return 0.5 * (1.0 + cos_sim)

def any_axis_orientation_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize if none of the body's principal axes align with the desired principal axes.

    The function computes the alignment between the corresponding principal axes (X, Y, Z) of the
    current end-effector orientation and the desired orientation. It takes the maximum absolute
    dot product across the three axes. The error is 1.0 minus this maximum alignment, meaning
    a lower error is achieved if at least one principal axis is well-aligned, regardless of direction.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # obtain the desired and current orientations in world frame
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]

    # Define standard basis vectors
    x_axis_base = torch.tensor([1.0, 0.0, 0.0], device=curr_quat_w.device, dtype=curr_quat_w.dtype)
    y_axis_base = torch.tensor([0.0, 1.0, 0.0], device=curr_quat_w.device, dtype=curr_quat_w.dtype)
    z_axis_base = torch.tensor([0.0, 0.0, 1.0], device=curr_quat_w.device, dtype=curr_quat_w.dtype)

    # Repeat for all environments
    x_axis_base = x_axis_base.repeat(curr_quat_w.shape[0], 1)
    y_axis_base = y_axis_base.repeat(curr_quat_w.shape[0], 1)
    z_axis_base = z_axis_base.repeat(curr_quat_w.shape[0], 1)

    # Apply quaternions to get rotated basis vectors
    curr_x = quat_apply(curr_quat_w, x_axis_base)
    curr_y = quat_apply(curr_quat_w, y_axis_base)
    curr_z = quat_apply(curr_quat_w, z_axis_base)

    des_x = quat_apply(des_quat_w, x_axis_base)
    des_y = quat_apply(des_quat_w, y_axis_base)
    des_z = quat_apply(des_quat_w, z_axis_base)

    # Calculate absolute dot products for corresponding axes
    align_xx = torch.abs(torch.sum(curr_x * des_x, dim=1))
    align_yy = torch.abs(torch.sum(curr_y * des_y, dim=1))
    align_zz = torch.abs(torch.sum(curr_z * des_z, dim=1))

    # Take the maximum alignment across the three axes
    max_alignment = torch.max(torch.max(align_xx, align_yy), align_zz)

    # The error is 1.0 minus the maximum alignment
    return 1.0 - max_alignment


def orientation_within_tolerance(
    env: ManagerBasedRLEnv, tolerance_deg: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward when orientation error is within a tolerance (in degrees)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    error = quat_error_magnitude(curr_quat_w, des_quat_w)
    tol = math.radians(tolerance_deg)
    return (error <= tol).to(error.dtype)

# def object_y_align_with_hand_z_reward(
#     env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, hand_tcp_body_name: str
# ) -> torch.Tensor:
#     """Reward for aligning the object's +y axis with the hand's +z axis."""
#     # Get object's +y axis in world frame
#     obj: RigidObject = env.scene[object_cfg.name]
#     object_quat_w = obj.data.root_quat_w
#     object_y_axis_local = torch.tensor([0.0, 1.0, 0.0], device=env.device, dtype=object_quat_w.dtype)
#     object_y_axis_world = quat_apply(object_quat_w, object_y_axis_local.repeat(env.num_envs, 1))

#     # Get hand's +z axis in world frame
#     robot: RigidObject = env.scene["robot"]
#     hand_body_idx = robot.data.body_names.index(hand_tcp_body_name)
#     hand_quat_w = robot.data.body_quat_w[:, hand_body_idx]
#     hand_z_axis_local = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=hand_quat_w.dtype)
#     hand_z_axis_world = quat_apply(hand_quat_w, hand_z_axis_local.repeat(env.num_envs, 1))

#     # Calculate dot product
#     # The dot product ranges from -1 (opposite) to 1 (aligned)
#     alignment_reward = torch.sum(object_y_axis_world * hand_z_axis_world, dim=1)

#     return alignment_reward
