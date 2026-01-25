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
Reward functions for the Grasp-v1 task.

The grasp task starts from a pre-grasp pose (TCP 3-6cm from cup) and must:
1. Fine-tune positioning to insert fingers around the cup
2. Close gripper to grasp
3. Lift slightly to confirm stable grasp
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _object_eef_distance(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Distance between the object and the specified end-effector link."""
    object_pos = env.scene[object_cfg.name].data.root_pos_w - env.scene.env_origins
    body_pos_w = env.scene["robot"].data.body_pos_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_pos = body_pos_w[:, eef_idx] - env.scene.env_origins
    return torch.norm(object_pos - eef_pos, dim=1)


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


def tcp_distance_to_cup(
    env: ManagerBasedRLEnv,
    tcp_body_name: str,
    target_cfg: SceneEntityCfg,
    offset: list[float],
) -> torch.Tensor:
    """Reward for TCP approaching the grasp point on the cup.

    Uses exponential reward: exp(-distance).
    """
    robot = env.scene["robot"]
    tcp_body_idx = robot.data.body_names.index(tcp_body_name)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_body_idx]

    target: RigidObject = env.scene[target_cfg.name]
    offset_tensor = torch.tensor(offset, device=env.device).unsqueeze(0)
    offset_tensor = offset_tensor.expand(target.data.root_quat_w.shape[0], -1)
    offset_world = quat_apply(target.data.root_quat_w, offset_tensor)
    target_pos_w = target.data.root_pos_w + offset_world

    dist = torch.norm(tcp_pos_w - target_pos_w, p=2, dim=-1)
    return torch.exp(-dist / 0.02)  # Sharp reward for precise positioning


def tcp_z_axis_to_target_alignment(
    env: ManagerBasedRLEnv,
    tcp_body_name: str,
    target_cfg: SceneEntityCfg,
    offset: list[float],
) -> torch.Tensor:
    """Reward for TCP's z-axis pointing towards the target."""
    robot = env.scene["robot"]
    tcp_body_idx = robot.data.body_names.index(tcp_body_name)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_body_idx]
    tcp_quat_w = robot.data.body_quat_w[:, tcp_body_idx]

    target: RigidObject = env.scene[target_cfg.name]
    offset_tensor = torch.tensor(offset, device=env.device).unsqueeze(0)
    offset_tensor = offset_tensor.expand(target.data.root_quat_w.shape[0], -1)
    offset_world = quat_apply(target.data.root_quat_w, offset_tensor)
    target_pos_w = target.data.root_pos_w + offset_world

    z_axis_tcp = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(tcp_pos_w)
    z_axis_world = quat_apply(tcp_quat_w, z_axis_tcp)

    dir_vec = target_pos_w - tcp_pos_w
    dir_norm = torch.norm(dir_vec, p=2, dim=-1, keepdim=True) + 1e-6
    dir_unit = dir_vec / dir_norm

    dot_product = torch.sum(z_axis_world * dir_unit, dim=-1)
    return dot_product


def grasp_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    reach_radius: float = 0.03,
    close_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward for grasping: close gripper when near object.

    Only rewards closing when the EEF is sufficiently close.
    """
    eef_dist = _object_eef_distance(env, eef_link_name, object_cfg)
    closure_amount = _hand_closure_amount(env, eef_link_name)

    dist_scale = 0.01
    close_scale = 0.15
    dist_score = torch.sigmoid((reach_radius - eef_dist) / dist_scale)
    close_score = torch.sigmoid((closure_amount - close_threshold) / close_scale)

    return dist_score * close_score


def finger_insertion_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    insertion_distance: float = 0.02,
) -> torch.Tensor:
    """Reward for successfully inserting fingers (EEF very close to object center).

    This encourages the robot to position the gripper so fingers are around the cup.
    """
    eef_dist = _object_eef_distance(env, eef_link_name, object_cfg)
    reward = torch.exp(-eef_dist / insertion_distance)
    return reward


def lift_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    lift_height: float = 0.05,
    table_height: float = 0.0,
) -> torch.Tensor:
    """Reward for lifting the object above table height."""
    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2] - (env.scene.env_origins[:, 2] + table_height)
    progress = height / lift_height
    return torch.clamp(progress, min=0.0, max=1.0)


def stable_grasp_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    lift_height: float = 0.03,
    grasp_distance: float = 0.03,
    close_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward for stable grasp: object lifted while gripper is closed and near object.

    This is a gated reward that only activates when all conditions are met.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    is_lifted = height > lift_height

    eef_dist = _object_eef_distance(env, eef_link_name, object_cfg)
    closure = _hand_closure_amount(env, eef_link_name)

    is_grasping = (eef_dist < grasp_distance) & (closure > close_threshold)

    return torch.where(is_lifted & is_grasping, 1.0, 0.0)


def gripper_open_penalty(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    close_distance: float = 0.03,
) -> torch.Tensor:
    """Penalty for keeping gripper open when very close to object.

    Encourages closing the gripper when in grasping range.
    """
    eef_dist = _object_eef_distance(env, eef_link_name, object_cfg)
    closure = _hand_closure_amount(env, eef_link_name)

    is_close = eef_dist < close_distance
    is_open = closure < 0.3

    return torch.where(is_close & is_open, -1.0, 0.0)


def hand_joint_position(
    env: ManagerBasedRLEnv,
    joint_name: str | list[str],
    target_pos: float,
) -> torch.Tensor:
    """Reward for gripper joint position being close to target."""
    robot = env.scene["robot"]
    if isinstance(joint_name, str):
        joint_idx = [robot.data.joint_names.index(joint_name)]
    else:
        joint_idx = [robot.data.joint_names.index(name) for name in joint_name]
    joint_pos = robot.data.joint_pos[:, joint_idx].mean(dim=-1)

    return torch.exp(-torch.abs(joint_pos - target_pos))
