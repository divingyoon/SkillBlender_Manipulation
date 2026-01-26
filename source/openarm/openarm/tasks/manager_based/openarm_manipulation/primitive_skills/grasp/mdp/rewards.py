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
    progress = torch.clamp(height / lift_height, min=0.0, max=1.0)
    return progress


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


def _is_gripper_in_contact(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    contact_distance: float = 0.02, # How close EEF needs to be to object
    min_closure: float = 0.7,      # How closed the gripper needs to be
) -> torch.Tensor:
    """
    Determines if the gripper is in "contact" with the object based on proximity and closure.
    """
    eef_dist = _object_eef_distance(env, eef_link_name, object_cfg)
    closure = _hand_closure_amount(env, eef_link_name)

    is_close = eef_dist < contact_distance
    is_closed_enough = closure > min_closure

    return is_close & is_closed_enough


def grasp_success_with_hold(
    env: ManagerBasedRLEnv,
    eef_link_name_left: str,
    eef_link_name_right: str,
    object_cfg_left: SceneEntityCfg,
    object_cfg_right: SceneEntityCfg,
    lift_threshold: float = 0.1,
    contact_distance: float = 0.02, # Passed to helper
    min_closure: float = 0.7,      # Passed to helper
) -> torch.Tensor:
    """
    Grasp success condition:
    1. Objects grasped (contact + gripper closed)
    2. Lifted above lift_threshold
    """
    # Current height
    cup1_height = env.scene[object_cfg_left.name].data.root_pos_w[:, 2]
    cup2_height = env.scene[object_cfg_right.name].data.root_pos_w[:, 2]

    # Use default_root_state for initial height (shape: [num_envs, 13] -> pos is [:, :3])
    initial_height_cup1 = env.scene[object_cfg_left.name].data.default_root_state[:, 2]
    initial_height_cup2 = env.scene[object_cfg_right.name].data.default_root_state[:, 2]

    # Lifted condition
    left_lifted = cup1_height > (initial_height_cup1 + lift_threshold)
    right_lifted = cup2_height > (initial_height_cup2 + lift_threshold)

    # Gripper contact condition
    left_contact = _is_gripper_in_contact(env, eef_link_name_left, object_cfg_left, contact_distance, min_closure)
    right_contact = _is_gripper_in_contact(env, eef_link_name_right, object_cfg_right, contact_distance, min_closure)

    # Success = lifted + contact maintained
    success = left_lifted & right_lifted & left_contact & right_contact

    return success.float()


def drop_penalty(
    env: ManagerBasedRLEnv,
    object_cfg_left: SceneEntityCfg,
    object_cfg_right: SceneEntityCfg,
    penalty_scale: float = -10.0,
    height_drop_threshold: float = 0.05, # 5cm drop threshold
) -> torch.Tensor:
    """
    Penalty for dropping the objects.
    - Penalizes if the height drops significantly after being lifted.
    """
    cup1_height = env.scene[object_cfg_left.name].data.root_pos_w[:, 2]
    cup2_height = env.scene[object_cfg_right.name].data.root_pos_w[:, 2]

    # Add attributes to environment for tracking max height per object per environment
    if not hasattr(env, "_max_height_cup1"):
        env._max_height_cup1 = torch.zeros(env.num_envs, device=env.device)
        env._max_height_cup2 = torch.zeros(env.num_envs, device=env.device)

    # Reset max heights for newly terminated/truncated environments
    reset_mask = (env.episode_length_buf == 0).squeeze(-1) # Assuming episode_length_buf is 0 at start of new episode

    env._max_height_cup1[reset_mask] = cup1_height[reset_mask]
    env._max_height_cup2[reset_mask] = cup2_height[reset_mask]

    # Update max heights for ongoing episodes
    env._max_height_cup1 = torch.maximum(env._max_height_cup1, cup1_height)
    env._max_height_cup2 = torch.maximum(env._max_height_cup2, cup2_height)

    # Calculate height drop from max observed height
    height_drop_cup1 = env._max_height_cup1 - cup1_height
    height_drop_cup2 = env._max_height_cup2 - cup2_height

    # Penalize if either cup drops below the threshold
    dropped = (height_drop_cup1 > height_drop_threshold) | (height_drop_cup2 > height_drop_threshold)

    return dropped.float() * penalty_scale


def continuous_hold_reward(
    env: ManagerBasedRLEnv,
    eef_link_name_left: str,
    eef_link_name_right: str,
    object_cfg_left: SceneEntityCfg,
    object_cfg_right: SceneEntityCfg,
    lift_threshold: float = 0.1,
    reward_scale: float = 1.0,
    contact_distance: float = 0.02, # Passed to helper
    min_closure: float = 0.7,      # Passed to helper
) -> torch.Tensor:
    """
    Reward for maintaining the lifted and grasped state.
    """
    cup1_height = env.scene[object_cfg_left.name].data.root_pos_w[:, 2]
    cup2_height = env.scene[object_cfg_right.name].data.root_pos_w[:, 2]

    # Use default_root_state for initial height
    initial_height_cup1 = env.scene[object_cfg_left.name].data.default_root_state[:, 2]
    initial_height_cup2 = env.scene[object_cfg_right.name].data.default_root_state[:, 2]

    # Both cups lifted above the threshold
    both_lifted = (cup1_height > initial_height_cup1 + lift_threshold) & \
                  (cup2_height > initial_height_cup2 + lift_threshold)

    # Both grippers in contact
    left_contact = _is_gripper_in_contact(env, eef_link_name_left, object_cfg_left, contact_distance, min_closure)
    right_contact = _is_gripper_in_contact(env, eef_link_name_right, object_cfg_right, contact_distance, min_closure)

    # Reward if both are lifted and in contact
    return (both_lifted & left_contact & right_contact).float() * reward_scale


# ============================================================================
# Phase-based rewards (grasp_2g style)
# ============================================================================

def _phase_weight(phase: torch.Tensor, weights: list[float], device: torch.device) -> torch.Tensor:
    """Select per-phase weights for each env."""
    weights_tensor = torch.tensor(weights, device=device)
    return weights_tensor[phase]


def _reach_success(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    reach_distance: float,
) -> torch.Tensor:
    dist = _object_eef_distance(env, eef_link_name, object_cfg)
    return dist < reach_distance


def _grasp_success(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    grasp_distance: float,
    close_threshold: float,
) -> torch.Tensor:
    dist = _object_eef_distance(env, eef_link_name, object_cfg)
    close = _hand_closure_amount(env, eef_link_name)
    return (dist < grasp_distance) & (close > close_threshold)


def _update_grasp_phase(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    lift_height: float,
    reach_distance: float,
    grasp_distance: float,
    close_threshold: float,
) -> torch.Tensor:
    """Update phase based on reach -> grasp -> lift conditions.

    Phase 0: Reaching
    Phase 1: Grasping (close enough, start closing)
    Phase 2: Lifting (grasped, lifting)
    Phase 3: Holding (lifted successfully)
    """
    if "left" in eef_link_name:
        phase_attr = "grasp_phase_left"
    elif "right" in eef_link_name:
        phase_attr = "grasp_phase_right"
    else:
        phase_attr = "grasp_phase"

    if not hasattr(env, phase_attr):
        setattr(env, phase_attr, torch.zeros(env.num_envs, device=env.device, dtype=torch.long))

    phase = getattr(env, phase_attr)
    if hasattr(env, "reset_buf"):
        phase = torch.where(env.reset_buf, torch.zeros_like(phase), phase)

    reach_ok = _reach_success(env, eef_link_name, object_cfg, reach_distance)
    grasp_ok = _grasp_success(env, eef_link_name, object_cfg, grasp_distance, close_threshold)
    obj: RigidObject = env.scene[object_cfg.name]
    lift_ok = obj.data.root_pos_w[:, 2] > lift_height

    phase = torch.where((phase == 0) & reach_ok, torch.tensor(1, device=env.device), phase)
    phase = torch.where((phase == 1) & grasp_ok, torch.tensor(2, device=env.device), phase)
    phase = torch.where((phase == 2) & lift_ok, torch.tensor(3, device=env.device), phase)
    setattr(env, phase_attr, phase)
    return phase


def object_ee_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    distance = _object_eef_distance(env, eef_link_name, object_cfg)
    return 1 - torch.tanh(distance / std)


def object_lift_progress(
    env: ManagerBasedRLEnv,
    lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Linear lift progress reward from table to target height."""
    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2]
    progress = height / lift_height
    return torch.clamp(progress, min=0.0, max=1.0)


def phase_reaching_reward(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Phase-gated reaching reward."""
    phase = _update_grasp_phase(
        env,
        eef_link_name,
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
    )
    reward = object_ee_distance_tanh(env, std, eef_link_name, object_cfg)
    return reward * _phase_weight(phase, phase_weights, env.device)


def phase_grasp_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    reach_radius: float,
    close_threshold: float,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Phase-gated grasp reward."""
    phase = _update_grasp_phase(
        env,
        eef_link_name,
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
    )
    reward = grasp_reward(env, eef_link_name, object_cfg, reach_radius, close_threshold)
    return reward * _phase_weight(phase, phase_weights, env.device)


def phase_lift_reward(
    env: ManagerBasedRLEnv,
    lift_height: float,
    object_cfg: SceneEntityCfg,
    phase_weights: list[float],
    phase_params: dict,
) -> torch.Tensor:
    """Phase-gated lift reward."""
    phase = _update_grasp_phase(
        env,
        phase_params["eef_link_name"],
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
    )
    reward = object_lift_progress(env, lift_height, object_cfg)
    return reward * _phase_weight(phase, phase_weights, env.device)


def grasp_phase_value(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    phase_params: dict,
) -> torch.Tensor:
    """Return the per-hand phase as a float for logging."""
    phase = _update_grasp_phase(
        env,
        phase_params["eef_link_name"],
        object_cfg,
        phase_params["lift_height"],
        phase_params["reach_distance"],
        phase_params["grasp_distance"],
        phase_params["close_threshold"],
    )
    return phase.to(torch.float32)
