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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _object_eef_distance(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Distance between the object and the specified end-effector link."""
    object_pos = env.scene[object_cfg.name].data.root_pos_w - env.scene.env_origins
    body_pos_w = env.scene["robot"].data.body_pos_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_pos = body_pos_w[:, eef_idx] - env.scene.env_origins
    return torch.norm(object_pos - eef_pos, dim=1)


def _hand_closure_amount(env: ManagerBasedRLEnv, eef_link_name: str) -> torch.Tensor:
    """Compute normalized closure amount for the hand associated with the given link."""
    # identify which hand is being used to access the appropriate action term
    if "left" in eef_link_name:
        hand_term = env.action_manager.get_term("left_hand_action")
    elif "right" in eef_link_name:
        hand_term = env.action_manager.get_term("right_hand_action")
    else:
        # fallback: no closure info if the link does not correspond to a hand
        return torch.zeros(env.num_envs, device=env.device)

    # processed_actions has shape (num_envs, num_gripper_joints); take mean across joints
    hand_action = hand_term.processed_actions  # current commanded positions

    # Determine default (open) finger position.
    if isinstance(hand_term._offset, torch.Tensor):
        default_pos = hand_term._offset.mean(dim=1)
    else:
        default_pos = torch.full((env.num_envs,), float(hand_term._offset), device=env.device)

    # Compute closure amount: positive when fingers are closing relative to default.
    mean_action = hand_action.mean(dim=1)
    return torch.clamp(
        (default_pos - mean_action) / (torch.abs(default_pos) + 1e-6), min=0.0, max=1.0
    )


def eef_to_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the end-effector being close to the object using tanh-kernel."""
    distance = _object_eef_distance(env, eef_link_name, object_cfg)
    return 1 - torch.tanh(distance / std)


def grasp_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Continuous reward encouraging the end effector to close onto the object.

    This function provides a smoother shaping signal than the previous binary
    reward. It combines two components:
    - Proximity component: encourages the end effector to approach the object.
    - Closure component: encourages the gripper fingers to close relative to
      their nominal open position.
    """
    # distance between end-effector and object
    eef_dist = _object_eef_distance(env, eef_link_name, object_cfg)

    closure_amount = _hand_closure_amount(env, eef_link_name)

    # Proximity factor: linear decay with distance. Reward is non-zero only within 0.05 m.
    reach_radius = 0.05
    proximity = torch.clamp(1.0 - (eef_dist / reach_radius), min=0.0, max=1.0)

    # Penalize closing when too far from the object.
    far_penalty = torch.where(eef_dist > reach_radius, -closure_amount, 0.0)

    # Final reward combines proximity-aligned closing and far-distance penalty.
    return proximity * closure_amount + far_penalty


def object_is_lifted_gated(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    eef_link_name: str,
    reach_radius: float,
    close_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward lifting only when the hand is close and sufficiently closed."""
    object: RigidObject = env.scene[object_cfg.name]
    lifted = object.data.root_pos_w[:, 2] > minimal_height

    eef_dist = _object_eef_distance(env, eef_link_name, object_cfg)
    closure_amount = _hand_closure_amount(env, eef_link_name)

    gated = (eef_dist < reach_radius) & (closure_amount > close_threshold)
    return torch.where(lifted & gated, 1.0, 0.0)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for lifting the object above a minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_is_held_gated(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    hold_duration: float,
    eef_link_name: str,
    reach_radius: float,
    close_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for holding the object above a height while grasp gate is satisfied."""
    object: RigidObject = env.scene[object_cfg.name]

    eef_dist = _object_eef_distance(env, eef_link_name, object_cfg)
    closure_amount = _hand_closure_amount(env, eef_link_name)

    gated = (eef_dist < reach_radius) & (closure_amount > close_threshold)
    is_lifted = object.data.root_pos_w[:, 2] > minimal_height
    should_hold = is_lifted & gated

    attr_name = f"hold_counter_{object_cfg.name}_{eef_link_name}"
    if not hasattr(env, attr_name):
        setattr(env, attr_name, torch.zeros(env.num_envs, device=env.device))

    hold_counter = getattr(env, attr_name)
    hold_counter = torch.where(
        should_hold,
        hold_counter + env.step_dt,
        torch.zeros_like(hold_counter),
    )
    setattr(env, attr_name, hold_counter)

    return torch.where(hold_counter > hold_duration, 1.0, 0.0)


def object_is_held(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    hold_duration: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for holding the object above a minimal height for a certain duration."""
    object: RigidObject = env.scene[object_cfg.name]

    if not hasattr(env, "hold_counter"):
        env.hold_counter = torch.zeros(env.num_envs, device=env.device)

    is_lifted = object.data.root_pos_w[:, 2] > minimal_height

    env.hold_counter = torch.where(
        is_lifted,
        env.hold_counter + env.step_dt,
        torch.zeros_like(env.hold_counter),
    )

    return torch.where(env.hold_counter > hold_duration, 1.0, 0.0)
