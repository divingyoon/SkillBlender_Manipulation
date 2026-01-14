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
    """Reward for grasping the object with a single hand."""
    # distance between eef and object
    eef_dist = _object_eef_distance(env, eef_link_name, object_cfg)

    # determine which hand is being used
    if "left" in eef_link_name:
        hand_term = env.action_manager.get_term("left_hand_action")
    elif "right" in eef_link_name:
        hand_term = env.action_manager.get_term("right_hand_action")
    else:
        # should not happen
        return torch.zeros_like(eef_dist)

    hand_action = hand_term.processed_actions
    if isinstance(hand_term._offset, torch.Tensor):
        default_pos = hand_term._offset.mean(dim=1)
    else:
        default_pos = torch.full((env.num_envs,), float(hand_term._offset), device=env.device)
    close_action = torch.mean(hand_action, dim=1) < (default_pos - 0.005)

    # reward is high when the gripper is closing and the distance is small
    reward = torch.where(
        (eef_dist < 0.1) & close_action,        1.0,
        0.0,
    )

    return reward


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for lifting the object above a minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


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

