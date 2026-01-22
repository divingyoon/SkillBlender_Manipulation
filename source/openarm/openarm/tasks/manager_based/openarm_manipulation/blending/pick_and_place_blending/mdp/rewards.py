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

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _object_eef_distance(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
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
    distance = _object_eef_distance(env, eef_link_name, object_cfg)
    return 1 - torch.tanh(distance / std)


def grasp_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    eef_dist = _object_eef_distance(env, eef_link_name, object_cfg)

    if "left" in eef_link_name:
        hand_action = env.action_manager.get_term("left_hand_action").raw_actions
    elif "right" in eef_link_name:
        hand_action = env.action_manager.get_term("right_hand_action").raw_actions
    else:
        return torch.zeros_like(eef_dist)

    close_action = torch.mean(hand_action, dim=1) > 0
    reward = torch.where((eef_dist < 0.1) & close_action, 1.0, 0.0)
    return reward


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_is_held(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    hold_duration: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
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


def handover_success(
    env: ManagerBasedRLEnv,
    eef_link_name_right: str,
    release_threshold: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Right hand grasps while left hand releases."""
    eef_dist = _object_eef_distance(env, eef_link_name_right, object_cfg)
    right_hand_action = env.action_manager.get_term("right_hand_action").raw_actions
    left_hand_action = env.action_manager.get_term("left_hand_action").raw_actions

    right_close = torch.mean(right_hand_action, dim=1) > 0
    left_release = torch.mean(left_hand_action, dim=1) <= release_threshold
    return torch.where((eef_dist < 0.1) & right_close & left_release, 1.0, 0.0)


def object_at_target(
    env: ManagerBasedRLEnv,
    target_pos: tuple[float, float, float],
    xy_tolerance: float = 0.06,
    max_height: float = 0.08,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    object_pos = object.data.root_pos_w - env.scene.env_origins
    target = torch.tensor(target_pos, device=object_pos.device)
    xy_dist = torch.norm(object_pos[:, :2] - target[:2], dim=1)
    height_ok = object_pos[:, 2] < max_height
    return torch.where((xy_dist < xy_tolerance) & height_ok, 1.0, 0.0)
