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
from openarm.tasks.manager_based.openarm_manipulation.bimanual.grasp_2g import mdp as grasp2g_mdp
import os

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../../../../")
)
_OBS_LOG_PATH = os.path.join(_ROOT_DIR, "obs_debug_pouring2.log")


def _append_obs_log(line: str) -> None:
    try:
        with open(_OBS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


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


def _object_eef_any_axis_alignment(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Maximum absolute alignment between any EE axis and any object axis."""
    object_quat = env.scene[object_cfg.name].data.root_quat_w
    body_quat_w = env.scene["robot"].data.body_quat_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_quat = body_quat_w[:, eef_idx]

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=env.device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device)
    x_axis = x_axis.repeat(env.num_envs, 1)
    y_axis = y_axis.repeat(env.num_envs, 1)
    z_axis = z_axis.repeat(env.num_envs, 1)

    eef_axes = [
        quat_apply(eef_quat, x_axis),
        quat_apply(eef_quat, y_axis),
        quat_apply(eef_quat, z_axis),
    ]
    obj_axes = [
        quat_apply(object_quat, x_axis),
        quat_apply(object_quat, y_axis),
        quat_apply(object_quat, z_axis),
    ]

    max_align = torch.zeros(env.num_envs, device=env.device)
    for eef_axis in eef_axes:
        for obj_axis in obj_axes:
            align = torch.abs(torch.sum(eef_axis * obj_axis, dim=1))
            max_align = torch.maximum(max_align, align)

    return max_align


def _hand_closure_amount(env: ManagerBasedRLEnv, eef_link_name: str) -> torch.Tensor:
    """Compute normalized closure amount for the hand associated with the given link."""
    if "left" in eef_link_name:
        hand_term = env.action_manager.get_term("left_hand_action")
    elif "right" in eef_link_name:
        hand_term = env.action_manager.get_term("right_hand_action")
    else:
        return torch.zeros(env.num_envs, device=env.device)

    # processed_actions has shape (num_envs, num_gripper_joints); take mean across joints
    hand_action = hand_term.processed_actions

    # Determine default (open) finger position.
    if isinstance(hand_term._offset, torch.Tensor):
        default_pos = hand_term._offset.mean(dim=1)
    else:
        default_pos = torch.full((env.num_envs,), float(hand_term._offset), device=env.device)

    # Positive when fingers close relative to default.
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
    distance = _object_eef_distance(env, eef_link_name, object_cfg)
    return 1 - torch.tanh(distance / std)


def tcp_x_axis_alignment(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Alignment between TCP x-axis and object's z-axis."""
    body_quat_w = env.scene["robot"].data.body_quat_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    axis_x = torch.tensor([1.0, 0.0, 0.0], device=env.device, dtype=body_quat_w.dtype)
    tcp_x_axis = quat_apply(body_quat_w[:, eef_idx], axis_x.expand(body_quat_w.shape[0], 3))

    object_quat_w = env.scene[object_cfg.name].data.root_quat_w
    axis_z = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=object_quat_w.dtype)
    object_z_axis = quat_apply(object_quat_w, axis_z.expand(object_quat_w.shape[0], 3))

    dot = torch.sum(tcp_x_axis * object_z_axis, dim=1)
    return torch.clamp(dot, min=0.0, max=1.0)


def phase_tcp_x_axis_alignment(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    hand: str = "left",
) -> torch.Tensor:
    """Phase-gated TCP x-axis alignment reward for a specific hand."""
    phase_min = _get_shared_phase(env)
    reward = tcp_x_axis_alignment(env, eef_link_name, object_cfg)
    return reward * (phase_min <= 1)


def grasp_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    eef_dist = _object_eef_distance(env, eef_link_name, object_cfg)
    closure_amount = _hand_closure_amount(env, eef_link_name)

    reach_radius = 0.05
    dist_scale = 0.03
    close_center = 0.6
    close_scale = 0.2
    dist_score = torch.sigmoid((reach_radius - eef_dist) / dist_scale)
    close_score = torch.sigmoid((closure_amount - close_center) / close_scale)

    return dist_score * close_score


def hand_open_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
) -> torch.Tensor:
    """Reward for keeping the hand open."""
    closure_amount = _hand_closure_amount(env, eef_link_name)
    return 1.0 - closure_amount


def phase_hand_open_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    hand: str = "left",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    off_distance: float = 0.10,
) -> torch.Tensor:
    """Phase-gated open-hand reward for a specific hand."""
    phase_min = _get_shared_phase(env)
    reward = hand_open_reward(env, eef_link_name)
    dist = _object_eef_distance(env, eef_link_name, object_cfg)
    far_enough = dist > off_distance
    return reward * (phase_min <= 1) * far_enough.to(reward.dtype)


def phase_object_goal_distance_with_ee(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    reach_std: float,
    hand: str,
    min_phase: int = 2,
) -> torch.Tensor:
    """Phase-gated goal tracking reward after lift/hold."""
    phase_min = _get_shared_phase(env)

    reward = grasp2g_mdp.object_goal_distance_with_ee(
        env,
        std=std,
        minimal_height=minimal_height,
        command_name=command_name,
        object_cfg=object_cfg,
        ee_frame_cfg=ee_frame_cfg,
        reach_std=reach_std,
    )
    return reward * (phase_min >= min_phase)


def phase_wrong_cup_penalty(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalty when a hand approaches the other cup during reach."""
    phase_min = _get_shared_phase(env)
    return _object_eef_distance(env, eef_link_name, object_cfg) * (phase_min == 0)


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


def hold_at_offset_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    offset_z: float = 0.2,
    grasp_distance: float = 0.02,
    close_threshold: float = 0.6,
    std: float = 0.05,
    hold_duration: float = 2.0,
    hold_distance: float = 0.05,
) -> torch.Tensor:
    """Reward holding the object at a fixed offset above its grasp position."""
    object_pos = env.scene[object_cfg.name].data.root_pos_w - env.scene.env_origins
    eef_dist = _object_eef_distance(env, eef_link_name, object_cfg)
    closure_amount = _hand_closure_amount(env, eef_link_name)
    grasp_ok = (eef_dist < grasp_distance) & (closure_amount > close_threshold)

    target_attr = f"hold_target_{object_cfg.name}"
    active_attr = f"hold_target_active_{object_cfg.name}"

    counter_attr = f"hold_target_counter_{object_cfg.name}"

    if not hasattr(env, target_attr):
        setattr(env, target_attr, torch.zeros_like(object_pos))
    if not hasattr(env, active_attr):
        setattr(env, active_attr, torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    if not hasattr(env, counter_attr):
        setattr(env, counter_attr, torch.zeros(env.num_envs, device=env.device))

    target_pos = getattr(env, target_attr)
    active = getattr(env, active_attr)
    counter = getattr(env, counter_attr)

    if hasattr(env, "reset_buf"):
        active = torch.where(env.reset_buf, torch.zeros_like(active), active)
        target_pos = torch.where(env.reset_buf.unsqueeze(1), torch.zeros_like(target_pos), target_pos)
        counter = torch.where(env.reset_buf, torch.zeros_like(counter), counter)

    new_targets = grasp_ok & (~active)
    if torch.any(new_targets):
        offset = torch.tensor([0.0, 0.0, offset_z], device=env.device, dtype=object_pos.dtype)
        target_pos = torch.where(new_targets.unsqueeze(1), object_pos + offset, target_pos)
        active = torch.where(new_targets, torch.ones_like(active), active)

    setattr(env, target_attr, target_pos)
    setattr(env, active_attr, active)

    if not torch.any(active):
        return torch.zeros(env.num_envs, device=env.device)

    dist = torch.norm(object_pos - target_pos, dim=1)
    within_hold = dist < hold_distance
    counter = torch.where(active & within_hold, counter + env.step_dt, torch.zeros_like(counter))
    setattr(env, counter_attr, counter)

    reward = 1.0 - torch.tanh(dist / std)
    sustained = counter >= hold_duration
    return reward * active.to(reward.dtype) * sustained.to(reward.dtype)


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


def cup_tipped_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_upright_dot: float = 0.5,
) -> torch.Tensor:
    """Penalty when the cup tilts beyond the upright threshold."""
    object_quat_w = env.scene[object_cfg.name].data.root_quat_w
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=object_quat_w.dtype)
    cup_z_axis = quat_apply(object_quat_w, z_axis.expand(object_quat_w.shape[0], 3))
    dot = torch.sum(cup_z_axis * z_axis, dim=1)
    return (dot < min_upright_dot).to(torch.float32)


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


def bead_to_target_distance_reward(
    env: ManagerBasedRLEnv,
    bead_name: str,
    target_name: str,
    std: float = 0.1,
) -> torch.Tensor:
    """Dense shaping reward for bringing the bead toward the target cup."""
    bead_pos = env.scene[bead_name].data.root_pos_w - env.scene.env_origins
    target_pos = env.scene[target_name].data.root_pos_w - env.scene.env_origins
    dist = torch.norm(bead_pos - target_pos, p=2, dim=-1)
    return 1.0 - torch.tanh(dist / std)


def bead_spill_penalty(
    env: ManagerBasedRLEnv,
    bead_name: str,
    target_name: str,
    min_height_offset: float = -0.02,
    xy_radius: float = 0.08,
) -> torch.Tensor:
    # Added for bead pouring stabilization: penalize bead spilling outside the target cup.
    """Penalty when the bead drops below the target cup height outside the cup area."""
    bead_pos = env.scene[bead_name].data.root_pos_w - env.scene.env_origins
    target_pos = env.scene[target_name].data.root_pos_w - env.scene.env_origins
    spill_height = target_pos[:, 2] + min_height_offset
    d_xy = torch.norm(bead_pos[:, :2] - target_pos[:, :2], p=2, dim=-1)
    spill = (bead_pos[:, 2] < spill_height) & (d_xy > xy_radius)
    return torch.where(spill, 1.0, 0.0)


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
    lift_height: float = 0.1,
    reach_distance: float = 0.18,
    align_threshold: float = 0.8,
    grasp_distance: float = 0.03,
    close_threshold: float = 0.2,
    hold_duration: float = 2.0,
    align_xy_threshold: float = 0.7,
    align_z_threshold: float = 0.7,
) -> torch.Tensor:
    if not hasattr(env, "pour_phase"):
        env.pour_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    if not hasattr(env, "pour_phase_left"):
        env.pour_phase_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    if not hasattr(env, "pour_phase_right"):
        env.pour_phase_right = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    align_xy = cup_xy_alignment(env, "object", "object2")
    align_z = cup_z_alignment(env, "object", "object2")
    align_ok = (align_xy > align_xy_threshold) & (align_z > align_z_threshold)

    left_phase = _update_pour_hand_phase(
        env,
        "openarm_left_ee_tcp",
        SceneEntityCfg("object"),
        lift_height,
        reach_distance,
        align_threshold,
        grasp_distance,
        close_threshold,
        hold_duration,
    )
    right_phase = _update_pour_hand_phase(
        env,
        "openarm_right_ee_tcp",
        SceneEntityCfg("object2"),
        lift_height,
        reach_distance,
        align_threshold,
        grasp_distance,
        close_threshold,
        hold_duration,
    )

    phase = env.pour_phase
    if hasattr(env, "reset_buf"):
        phase = torch.where(env.reset_buf, torch.zeros_like(phase), phase)
    both_ready = (left_phase >= 3) & (right_phase >= 3)
    # Group phases: 4 = Transfer, 5 = Pouring.
    phase = torch.where((phase == 0) & both_ready, torch.tensor(4, device=env.device), phase)
    phase = torch.where((phase == 4) & align_ok, torch.tensor(5, device=env.device), phase)
    env.pour_phase = phase
    if not hasattr(env, "_phase_print_last"):
        env._phase_print_last = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    if hasattr(env, "common_step_counter") and env.common_step_counter % 100 == 0:
        if env._phase_print_last[0].item() != phase[0].item():
            print(f"[INFO] Pour phase (env0): {int(phase[0].item())}")
            env._phase_print_last[0] = phase[0]
    return phase


def _get_shared_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Shared phase based on the slower hand."""
    _update_pour_phase(env)
    env.pour_phase_min = torch.minimum(env.pour_phase_left, env.pour_phase_right)
    return env.pour_phase_min


def phase_left_value(env: ManagerBasedRLEnv) -> torch.Tensor:
    _update_pour_phase(env)
    return env.pour_phase_left.to(dtype=torch.float32)


def phase_right_value(env: ManagerBasedRLEnv) -> torch.Tensor:
    _update_pour_phase(env)
    return env.pour_phase_right.to(dtype=torch.float32)


def phase_shared_value(env: ManagerBasedRLEnv) -> torch.Tensor:
    phase_min = _get_shared_phase(env)
    return phase_min.to(dtype=torch.float32)


def _update_pour_hand_phase(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    lift_height: float,
    reach_distance: float,
    align_threshold: float,
    grasp_distance: float,
    close_threshold: float,
    hold_duration: float,
) -> torch.Tensor:
    """Mirror grasp_2g phase logic for per-hand reach -> grasp -> lift -> hold."""
    if "left" in eef_link_name:
        phase_attr = "pour_phase_left"
    elif "right" in eef_link_name:
        phase_attr = "pour_phase_right"
    else:
        phase_attr = "pour_phase"

    if not hasattr(env, phase_attr):
        setattr(env, phase_attr, torch.zeros(env.num_envs, device=env.device, dtype=torch.long))

    phase = getattr(env, phase_attr)
    if hasattr(env, "reset_buf"):
        phase = torch.where(env.reset_buf, torch.zeros_like(phase), phase)

    dist = _object_eef_distance(env, eef_link_name, object_cfg)
    align = tcp_x_axis_alignment(env, eef_link_name, object_cfg)
    reach_ok = (dist < reach_distance) & (align > align_threshold)

    close = _hand_closure_amount(env, eef_link_name)
    grasp_ok = (dist < grasp_distance) & (close > close_threshold)

    obj: RigidObject = env.scene[object_cfg.name]
    lift_ok = obj.data.root_pos_w[:, 2] > lift_height

    if hasattr(env, "common_step_counter") and env.common_step_counter % 200 == 0:
        try:
            idx = 0
            phase_val = int(phase[idx].item())
            dist_val = float(dist[idx].item())
            align_val = float(align[idx].item())
            close_val = float(close[idx].item())
            lift_val = float(obj.data.root_pos_w[idx, 2].item())
            _append_obs_log(
                f"[PHASE_DBG] hand={eef_link_name} phase={phase_val} "
                f"dist={dist_val:.4f} align={align_val:.4f} "
                f"close={close_val:.4f} lift_z={lift_val:.4f}"
            )
        except Exception:
            pass

    phase = torch.where((phase == 0) & reach_ok, torch.tensor(1, device=env.device), phase)
    phase = torch.where((phase == 1) & grasp_ok, torch.tensor(2, device=env.device), phase)
    phase = torch.where((phase == 2) & lift_ok, torch.tensor(3, device=env.device), phase)
    setattr(env, phase_attr, phase)
    return phase


def phase_grasp_reward(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    phase_min = _get_shared_phase(env)
    reward = grasp_reward(env, eef_link_name, object_cfg)
    return reward * (phase_min == 1)


def phase_reach_reward(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Phase-gated reach reward before grasp."""
    phase_min = _get_shared_phase(env)
    reward = eef_to_object_distance(env, std, eef_link_name, object_cfg)
    return reward * (phase_min == 0)


def phase_lift_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    phase_min = _get_shared_phase(env)
    reward = object_is_lifted(env, minimal_height, object_cfg)
    return reward * (phase_min == 2)


def phase_hold_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    hold_duration: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    phase_min = _get_shared_phase(env)
    reward = _cup_is_held(env, minimal_height, hold_duration, object_cfg)
    return reward * (phase_min == 3)


def phase_cup_xy_alignment(
    env: ManagerBasedRLEnv,
    source_name: str,
    target_name: str,
) -> torch.Tensor:
    phase = _update_pour_phase(env)
    reward = cup_xy_alignment(env, source_name, target_name)
    return reward * (phase == 4)


def phase_cup_z_alignment(
    env: ManagerBasedRLEnv,
    source_name: str,
    target_name: str,
) -> torch.Tensor:
    phase = _update_pour_phase(env)
    reward = cup_z_alignment(env, source_name, target_name)
    return reward * (phase == 4)


def phase_cup_tilt_reward(
    env: ManagerBasedRLEnv,
    source_name: str,
    target_name: str,
) -> torch.Tensor:
    phase = _update_pour_phase(env)
    reward = cup_tilt_reward(env, source_name, target_name)
    return reward * (phase == 5)


def phase_bead_in_target(
    env: ManagerBasedRLEnv,
    bead_name: str,
    target_name: str,
    radius: float = 0.1,
) -> torch.Tensor:
    phase = _update_pour_phase(env)
    reward = bead_in_target_reward(env, bead_name, target_name, radius)
    return reward * (phase == 5)


def phase_bead_to_target_distance(
    env: ManagerBasedRLEnv,
    bead_name: str,
    target_name: str,
    std: float = 0.1,
) -> torch.Tensor:
    phase = _update_pour_phase(env)
    reward = bead_to_target_distance_reward(env, bead_name, target_name, std)
    return reward * (phase == 5)


def phase_bead_spill_penalty(
    env: ManagerBasedRLEnv,
    bead_name: str,
    target_name: str,
    min_height_offset: float = -0.02,
    xy_radius: float = 0.08,
) -> torch.Tensor:
    phase = _update_pour_phase(env)
    penalty = bead_spill_penalty(env, bead_name, target_name, min_height_offset, xy_radius)
    return penalty * (phase == 5)
