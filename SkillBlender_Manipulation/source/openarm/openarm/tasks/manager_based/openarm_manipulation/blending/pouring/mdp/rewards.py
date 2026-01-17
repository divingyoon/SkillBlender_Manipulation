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
from isaaclab.utils.math import quat_apply

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


def cup_xy_alignment(
    env: ManagerBasedRLEnv,
    source_name: str,
    target_name: str,
    target_d_xy: float = 0.095,
) -> torch.Tensor:
    """Reward for aligning cup positions in the XY plane."""
    source_pos = env.scene[source_name].data.root_pos_w - env.scene.env_origins
    target_pos = env.scene[target_name].data.root_pos_w - env.scene.env_origins
    d_xy = torch.norm(source_pos[:, :2] - target_pos[:, :2], p=2, dim=-1)
    distance_far = torch.relu(d_xy - target_d_xy)
    penalty_tan = torch.tanh(distance_far * 20.5) * 0.3
    penalty_linear = distance_far * 2.4
    reward = torch.where(d_xy <= target_d_xy, 1.0, 1.0 - penalty_tan - penalty_linear)
    return reward.clamp_min(0.0)


def cup_z_alignment(
    env: ManagerBasedRLEnv,
    source_name: str,
    target_name: str,
    target_d_z_min: float = 0.11,
    target_d_z_max: float = 0.12,
) -> torch.Tensor:
    """Reward for aligning cup heights within a target band."""
    source_pos = env.scene[source_name].data.root_pos_w - env.scene.env_origins
    target_pos = env.scene[target_name].data.root_pos_w - env.scene.env_origins
    d_z = source_pos[:, 2] - target_pos[:, 2]
    distance_close = torch.relu(-d_z + target_d_z_min) * 25
    distance_far = torch.relu(d_z - target_d_z_max) * 25
    penalty_tan_close = torch.tanh(distance_close * 20.5) * 0.3
    penalty_linear_close = distance_close * 0.7
    penalty_tan_far = torch.tanh(distance_far * 20.5) * 0.3
    penalty_linear_far = distance_far * 0.7
    in_range = (d_z >= target_d_z_min) & (d_z <= target_d_z_max)
    reward = torch.where(
        in_range,
        1.0,
        1.0 - penalty_tan_close - penalty_linear_close - penalty_tan_far - penalty_linear_far,
    )
    return reward.clamp_min(0.0)


def cup_tilt_reward(
    env: ManagerBasedRLEnv,
    source_name: str,
    target_name: str,
) -> torch.Tensor:
    """Reward for tilting the source cup toward the target cup."""
    source_pos = env.scene[source_name].data.root_pos_w - env.scene.env_origins
    target_pos = env.scene[target_name].data.root_pos_w - env.scene.env_origins
    source_quat = env.scene[source_name].data.root_quat_w
    target_vector = target_pos - source_pos
    target_vector = torch.nn.functional.normalize(target_vector, p=2.0, dim=-1)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=source_pos.device, dtype=source_pos.dtype).expand_as(source_pos)
    cup_z_axis = quat_apply(source_quat, z_axis)
    cup_z_axis = torch.nn.functional.normalize(cup_z_axis, p=2.0, dim=-1)
    dot_product = torch.sum(cup_z_axis * target_vector, dim=-1).clamp(-1.0, 1.0)
    theta = torch.acos(dot_product)
    return (1.0 - torch.tanh(theta * 0.65)).clamp_min(0.0)


def bead_in_target_reward(
    env: ManagerBasedRLEnv,
    bead_name: str,
    target_name: str,
    radius: float = 0.1,
) -> torch.Tensor:
    """Reward when bead is close to the target cup."""
    bead_pos = env.scene[bead_name].data.root_pos_w - env.scene.env_origins
    target_pos = env.scene[target_name].data.root_pos_w - env.scene.env_origins
    dist = torch.norm(bead_pos - target_pos, p=2, dim=-1)
    return torch.where(dist <= radius, 1.0, 0.0)


def _cup_is_held(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    hold_duration: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    if not hasattr(env, "cup_hold_counters"):
        env.cup_hold_counters = {}
    name = object_cfg.name
    if name not in env.cup_hold_counters:
        env.cup_hold_counters[name] = torch.zeros(env.num_envs, device=env.device)

    obj: RigidObject = env.scene[name]
    is_lifted = obj.data.root_pos_w[:, 2] > minimal_height
    env.cup_hold_counters[name] = torch.where(
        is_lifted,
        env.cup_hold_counters[name] + env.step_dt,
        torch.zeros_like(env.cup_hold_counters[name]),
    )
    return torch.where(env.cup_hold_counters[name] > hold_duration, 1.0, 0.0)


def _update_pour_phase(
    env: ManagerBasedRLEnv,
    min_height: float = 0.15,
    hold_duration: float = 5.0,
    align_xy_threshold: float = 0.7,
    align_z_threshold: float = 0.7,
) -> torch.Tensor:
    if not hasattr(env, "pour_phase"):
        env.pour_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    left_grasp = grasp_reward(env, "openarm_left_ee_tcp", SceneEntityCfg("object")) > 0.5
    right_grasp = grasp_reward(env, "openarm_right_ee_tcp", SceneEntityCfg("object2")) > 0.5
    grasp_ok = left_grasp & right_grasp

    left_lift = object_is_lifted(env, min_height, SceneEntityCfg("object")) > 0.5
    right_lift = object_is_lifted(env, min_height, SceneEntityCfg("object2")) > 0.5
    lift_ok = left_lift & right_lift

    left_hold = _cup_is_held(env, min_height, hold_duration, SceneEntityCfg("object")) > 0.5
    right_hold = _cup_is_held(env, min_height, hold_duration, SceneEntityCfg("object2")) > 0.5
    hold_ok = left_hold & right_hold

    align_xy = cup_xy_alignment(env, "object", "object2")
    align_z = cup_z_alignment(env, "object", "object2")
    align_ok = (align_xy > align_xy_threshold) & (align_z > align_z_threshold)

    phase = env.pour_phase
    phase = torch.where((phase == 0) & grasp_ok, torch.tensor(1, device=env.device), phase)
    phase = torch.where((phase == 1) & lift_ok, torch.tensor(2, device=env.device), phase)
    phase = torch.where((phase == 2) & hold_ok, torch.tensor(3, device=env.device), phase)
    phase = torch.where((phase == 3) & align_ok, torch.tensor(4, device=env.device), phase)
    env.pour_phase = phase
    if not hasattr(env, "_phase_print_last"):
        env._phase_print_last = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    if hasattr(env, "common_step_counter") and env.common_step_counter % 100 == 0:
        if env._phase_print_last[0].item() != phase[0].item():
            print(f"[INFO] Pour phase (env0): {int(phase[0].item())}")
            env._phase_print_last[0] = phase[0]
    return phase


def phase_grasp_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    phase = _update_pour_phase(env)
    reward = grasp_reward(env, eef_link_name, object_cfg)
    return reward * (phase == 0)


def phase_lift_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    phase = _update_pour_phase(env)
    reward = object_is_lifted(env, minimal_height, object_cfg)
    return reward * (phase == 1)


def phase_hold_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    hold_duration: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    phase = _update_pour_phase(env)
    reward = _cup_is_held(env, minimal_height, hold_duration, object_cfg)
    return reward * (phase == 2)


def phase_cup_xy_alignment(
    env: ManagerBasedRLEnv,
    source_name: str,
    target_name: str,
) -> torch.Tensor:
    phase = _update_pour_phase(env)
    reward = cup_xy_alignment(env, source_name, target_name)
    return reward * (phase == 3)


def phase_cup_z_alignment(
    env: ManagerBasedRLEnv,
    source_name: str,
    target_name: str,
) -> torch.Tensor:
    phase = _update_pour_phase(env)
    reward = cup_z_alignment(env, source_name, target_name)
    return reward * (phase == 3)


def phase_cup_tilt_reward(
    env: ManagerBasedRLEnv,
    source_name: str,
    target_name: str,
) -> torch.Tensor:
    phase = _update_pour_phase(env)
    reward = cup_tilt_reward(env, source_name, target_name)
    return reward * (phase == 3)


def phase_bead_in_target(
    env: ManagerBasedRLEnv,
    bead_name: str,
    target_name: str,
    radius: float = 0.1,
) -> torch.Tensor:
    phase = _update_pour_phase(env)
    reward = bead_in_target_reward(env, bead_name, target_name, radius)
    return reward * (phase >= 4)
