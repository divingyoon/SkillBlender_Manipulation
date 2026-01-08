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

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

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


def position_command_error_with_deadzone(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, threshold: float = 0.01
) -> torch.Tensor:
    """Penalize tracking error but ignore small errors within a deadzone."""
    error = position_command_error(env, command_name, asset_cfg)
    return torch.where(error < threshold, torch.zeros_like(error), error)


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


def orientation_x_axis_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize misalignment of the body x-axis with the desired x-axis."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=curr_quat_w.device, dtype=curr_quat_w.dtype)
    x_axis = x_axis.repeat(curr_quat_w.shape[0], 1)
    curr_x = quat_apply(curr_quat_w, x_axis)
    des_x = quat_apply(des_quat_w, x_axis)
    cos_sim = torch.sum(curr_x * des_x, dim=1)
    return 1.0 - cos_sim


def near_goal_joint_vel_l2(
    env: ManagerBasedRLEnv,
    left_command_name: str = "left_ee_pose",
    right_command_name: str = "right_ee_pose",
    pos_threshold: float = 0.01,
    ori_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ll_dg_ee"]),
    right_ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["rl_dg_ee"]),
) -> torch.Tensor:
    """Penalize joint velocity only when both end-effectors are at the goal pose."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        joint_ids = slice(None)

    left_cmd = env.command_manager.get_command(left_command_name)
    right_cmd = env.command_manager.get_command(right_command_name)

    left_des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, left_cmd[:, :3])
    right_des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, right_cmd[:, :3])

    left_des_quat_w = quat_mul(robot.data.root_quat_w, left_cmd[:, 3:7])
    right_des_quat_w = quat_mul(robot.data.root_quat_w, right_cmd[:, 3:7])

    left_body_id = left_ee_cfg.body_ids[0]
    right_body_id = right_ee_cfg.body_ids[0]

    left_pos_err = torch.norm(robot.data.body_pos_w[:, left_body_id] - left_des_pos_w, dim=1)
    right_pos_err = torch.norm(robot.data.body_pos_w[:, right_body_id] - right_des_pos_w, dim=1)

    left_ori_err = quat_error_magnitude(robot.data.body_quat_w[:, left_body_id], left_des_quat_w)
    right_ori_err = quat_error_magnitude(robot.data.body_quat_w[:, right_body_id], right_des_quat_w)

    near_goal = (left_pos_err < pos_threshold) & (right_pos_err < pos_threshold)
    near_goal &= (left_ori_err < ori_threshold) & (right_ori_err < ori_threshold)

    vel_penalty = torch.sum(torch.square(robot.data.joint_vel[:, joint_ids]), dim=1)
    return torch.where(near_goal, vel_penalty, torch.zeros_like(vel_penalty))


def _smooth_gate(value: torch.Tensor, on: float, off: float) -> torch.Tensor:
    """Return a [0, 1] gate that ramps down as value goes from on -> off."""
    denom = max(off - on, 1e-6)
    return torch.clamp((off - value) / denom, min=0.0, max=1.0)


def _scale_gate(gate: torch.Tensor, min_scale: float) -> torch.Tensor:
    """Blend gate with a minimum scale to keep a small penalty active."""
    if min_scale <= 0.0:
        return gate
    return min_scale + (1.0 - min_scale) * gate


def _bimanual_pose_errors(
    env: ManagerBasedRLEnv,
    left_command_name: str,
    right_command_name: str,
    robot_cfg: SceneEntityCfg,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute position/orientation errors for both end-effectors."""
    robot: Articulation = env.scene[robot_cfg.name]

    left_cmd = env.command_manager.get_command(left_command_name)
    right_cmd = env.command_manager.get_command(right_command_name)

    left_des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, left_cmd[:, :3])
    right_des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, right_cmd[:, :3])

    left_des_quat_w = quat_mul(robot.data.root_quat_w, left_cmd[:, 3:7])
    right_des_quat_w = quat_mul(robot.data.root_quat_w, right_cmd[:, 3:7])

    left_body_id = left_ee_cfg.body_ids[0]
    right_body_id = right_ee_cfg.body_ids[0]

    left_pos_err = torch.norm(robot.data.body_pos_w[:, left_body_id] - left_des_pos_w, dim=1)
    right_pos_err = torch.norm(robot.data.body_pos_w[:, right_body_id] - right_des_pos_w, dim=1)

    left_ori_err = quat_error_magnitude(robot.data.body_quat_w[:, left_body_id], left_des_quat_w)
    right_ori_err = quat_error_magnitude(robot.data.body_quat_w[:, right_body_id], right_des_quat_w)

    return left_pos_err, right_pos_err, left_ori_err, right_ori_err


def gated_orientation_z_axis_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    pos_on: float = 0.015,
    pos_off: float = 0.05,
    min_scale: float = 0.0,
) -> torch.Tensor:
    """Penalize z-axis misalignment only when close to the target position."""
    pos_err = position_command_error(env, command_name, asset_cfg)
    gate = _scale_gate(_smooth_gate(pos_err, pos_on, pos_off), min_scale)
    return orientation_z_axis_error(env, command_name, asset_cfg) * gate


def gated_orientation_x_axis_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    pos_on: float = 0.015,
    pos_off: float = 0.05,
    min_scale: float = 0.0,
) -> torch.Tensor:
    """Penalize x-axis misalignment only when close to the target position."""
    pos_err = position_command_error(env, command_name, asset_cfg)
    gate = _scale_gate(_smooth_gate(pos_err, pos_on, pos_off), min_scale)
    return orientation_x_axis_error(env, command_name, asset_cfg) * gate


def gated_action_rate_l2(
    env: ManagerBasedRLEnv,
    left_command_name: str,
    right_command_name: str,
    pos_on: float = 0.008,
    pos_off: float = 0.02,
    ori_on: float = 0.03,
    ori_off: float = 0.1,
    min_scale: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ll_dg_ee"]),
    right_ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["rl_dg_ee"]),
) -> torch.Tensor:
    """Penalize action rate more when both end-effectors are near the goal pose."""
    left_pos_err, right_pos_err, left_ori_err, right_ori_err = _bimanual_pose_errors(
        env, left_command_name, right_command_name, robot_cfg, left_ee_cfg, right_ee_cfg
    )
    pos_gate = torch.minimum(_smooth_gate(left_pos_err, pos_on, pos_off), _smooth_gate(right_pos_err, pos_on, pos_off))
    ori_gate = torch.minimum(_smooth_gate(left_ori_err, ori_on, ori_off), _smooth_gate(right_ori_err, ori_on, ori_off))
    gate = _scale_gate(pos_gate * ori_gate, min_scale)
    return base_mdp.action_rate_l2(env) * gate


def gated_joint_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    ee_cfg: SceneEntityCfg,
    pos_on: float = 0.01,
    pos_off: float = 0.03,
    ori_on: float = 0.03,
    ori_off: float = 0.1,
    min_scale: float = 0.0,
) -> torch.Tensor:
    """Penalize joint velocity when the corresponding end-effector is near the goal pose."""
    pos_err = position_command_error(env, command_name, ee_cfg)
    ori_err = orientation_command_error(env, command_name, ee_cfg)
    gate = _smooth_gate(pos_err, pos_on, pos_off) * _smooth_gate(ori_err, ori_on, ori_off)
    gate = _scale_gate(gate, min_scale)
    return base_mdp.joint_vel_l2(env, asset_cfg=asset_cfg) * gate


def gated_bimanual_joint_vel_l2(
    env: ManagerBasedRLEnv,
    left_command_name: str,
    right_command_name: str,
    pos_on: float = 0.008,
    pos_off: float = 0.02,
    ori_on: float = 0.03,
    ori_off: float = 0.1,
    min_scale: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ll_dg_ee"]),
    right_ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["rl_dg_ee"]),
) -> torch.Tensor:
    """Penalize overall joint velocity when both end-effectors are near the goal pose."""
    left_pos_err, right_pos_err, left_ori_err, right_ori_err = _bimanual_pose_errors(
        env, left_command_name, right_command_name, asset_cfg, left_ee_cfg, right_ee_cfg
    )
    pos_gate = torch.minimum(_smooth_gate(left_pos_err, pos_on, pos_off), _smooth_gate(right_pos_err, pos_on, pos_off))
    ori_gate = torch.minimum(_smooth_gate(left_ori_err, ori_on, ori_off), _smooth_gate(right_ori_err, ori_on, ori_off))
    gate = _scale_gate(pos_gate * ori_gate, min_scale)
    return base_mdp.joint_vel_l2(env, asset_cfg=asset_cfg) * gate

