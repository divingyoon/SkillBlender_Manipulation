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
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _finger_contact_flags_from_sensor(
    force_magnitudes: torch.Tensor,
    contact_threshold: float,
    sensor_body_names: list[str] | tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Aggregate link-level contact forces into finger-level boolean flags.

    For the T3 left hand, preferred mapping is by sensor link names:
    - finger_1: seg3, tip
    - finger_2/3/4: seg2, seg3, tip
    - finger_5: seg3, tip
    plus 3 palm sensors (ignored for finger coverage).
    """
    num_links = force_magnitudes.shape[1]
    link_flags = force_magnitudes > contact_threshold

    # 1) Preferred: explicit mapping by sensor link names when available.
    if sensor_body_names is not None and len(sensor_body_names) == num_links:
        finger_flags: list[torch.Tensor] = []
        for finger_id in (1, 2, 3, 4, 5):
            idxs = [i for i, name in enumerate(sensor_body_names) if f"finger_{finger_id}_" in str(name)]
            if idxs:
                finger_flags.append(link_flags[:, idxs].any(dim=1))
            else:
                finger_flags.append(torch.zeros(link_flags.shape[0], device=link_flags.device, dtype=torch.bool))
        return torch.stack(finger_flags, dim=1)

    # 2) Fallback for known T3 ordering without names:
    # [palm1, palm2, palm3, f1(2), f2(3), f3(3), f4(3), f5(2)] = 16
    if num_links == 16:
        groups = [
            link_flags[:, 3:5],    # finger 1
            link_flags[:, 5:8],    # finger 2
            link_flags[:, 8:11],   # finger 3
            link_flags[:, 11:14],  # finger 4
            link_flags[:, 14:16],  # finger 5
        ]
        return torch.stack([g.any(dim=1) for g in groups], dim=1)

    # 3) Older compact setup: 10 links -> pairwise mapping.
    if num_links >= 10:
        trimmed = link_flags[:, :10]
        finger_flags = trimmed.reshape(trimmed.shape[0], 5, 2).any(dim=2)
    else:
        # Last-resort fallback: contiguous chunks into up to 5 groups.
        group_count = max(1, min(5, num_links))
        chunk = max(1, num_links // group_count)
        groups: list[torch.Tensor] = []
        for i in range(group_count):
            s = i * chunk
            e = num_links if i == group_count - 1 else min(num_links, (i + 1) * chunk)
            if s >= num_links:
                groups.append(torch.zeros(link_flags.shape[0], device=link_flags.device, dtype=torch.bool))
            else:
                groups.append(link_flags[:, s:e].any(dim=1))
        finger_flags = torch.stack(groups, dim=1)

    return finger_flags


def _select_sensor_body_names(
    sensor_body_names: list[str] | tuple[str, ...] | None,
    body_ids,
) -> list[str] | tuple[str, ...] | None:
    """Select sensor body names using body_ids that may be slice/list/tensor."""
    if sensor_body_names is None or body_ids is None:
        return sensor_body_names
    if isinstance(body_ids, slice):
        return sensor_body_names[body_ids]
    if torch.is_tensor(body_ids):
        body_ids = body_ids.tolist()
    return [sensor_body_names[int(i)] for i in body_ids]


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w[:, :3]
    obj_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, obj_pos_w)
    return obj_pos_b


def _get_episode_initial_object_xy(
    env: ManagerBasedRLEnv,
    obj: RigidObject,
    cache_attr: str,
) -> torch.Tensor:
    """Track per-episode initial object XY in world frame."""
    current_xy = obj.data.root_pos_w[:, :2]
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, current_xy.clone())
    initial_xy = getattr(env, cache_attr)
    reset_mask = (env.episode_length_buf == 0).squeeze(-1)
    initial_xy[reset_mask] = current_xy[reset_mask]
    setattr(env, cache_attr, initial_xy)
    return initial_xy


def _compute_grasp_target_pos_w(
    env: ManagerBasedRLEnv,
    obj: RigidObject,
    ee_pos_w: torch.Tensor,
    use_dynamic_z: bool,
    dynamic_z_state_attr: str | None = None,
) -> torch.Tensor:
    """Compute grasp target from grasp2g offset in world frame."""
    cfg = getattr(env, "cfg", None)
    obj_pos_w = obj.data.root_pos_w.clone()
    base_offset = getattr(cfg, "grasp2g_target_offset", (0.0, 0.0, 0.08))

    if not (isinstance(base_offset, (list, tuple)) and len(base_offset) == 3):
        return obj_pos_w

    offset_xy_local = torch.zeros(obj_pos_w.shape[0], 3, device=obj_pos_w.device, dtype=obj_pos_w.dtype)
    offset_xy_local[:, 0] = base_offset[0]
    offset_xy_local[:, 1] = base_offset[1]
    offset_xy_w = quat_apply(obj.data.root_quat_w, offset_xy_local)
    target_xy_w = obj_pos_w[:, :2] + offset_xy_w[:, :2]

    z_value: torch.Tensor | float = base_offset[2]
    if use_dynamic_z:
        # Use XY distance to the offset target (not cup center) for z transition.
        xy_dist = torch.norm(ee_pos_w[:, :2] - target_xy_w, dim=1)
        z_high = float(getattr(cfg, "reach_dynamic_z_high", 0.2))
        x_hi = float(getattr(cfg, "reach_dynamic_xy_hi", 0.06))
        x_lo = float(getattr(cfg, "reach_dynamic_xy_lo", 0.015))
        x_gate = float(getattr(cfg, "reach_dynamic_xy_gate", x_hi))
        x_gate = min(x_gate, x_hi)
        x_gate = max(x_gate, x_lo + 1e-6)
        x_hi = max(x_hi, x_lo + 1e-6)

        # Keep high-Z approach until XY is close enough, then start descending.
        u = torch.clamp((x_gate - xy_dist) / (x_gate - x_lo), 0.0, 1.0)
        u = u * u * (3.0 - 2.0 * u)  # smoothstep
        z_value_raw = z_high * (1.0 - u) + float(base_offset[2]) * u

        # Limit per-step Z descent speed to avoid sudden dives toward the cup.
        descent_rate = float(getattr(cfg, "reach_dynamic_z_descent_rate", 0.0))
        if descent_rate > 0.0 and dynamic_z_state_attr is not None:
            if not hasattr(env, dynamic_z_state_attr):
                setattr(env, dynamic_z_state_attr, torch.full_like(z_value_raw, z_high))
            z_prev = getattr(env, dynamic_z_state_attr)
            reset_mask = (env.episode_length_buf == 0).squeeze(-1)
            z_prev[reset_mask] = z_high

            z_floor = z_prev - descent_rate
            z_value = torch.maximum(z_value_raw, z_floor)
            setattr(env, dynamic_z_state_attr, z_value)
        else:
            z_value = z_value_raw

    offset_local = torch.zeros(obj_pos_w.shape[0], 3, device=obj_pos_w.device, dtype=obj_pos_w.dtype)
    offset_local[:, 0] = base_offset[0]
    offset_local[:, 1] = base_offset[1]
    if isinstance(z_value, torch.Tensor):
        offset_local[:, 2] = z_value
    else:
        offset_local[:, 2] = float(z_value)
    offset_w = quat_apply(obj.data.root_quat_w, offset_local)
    return obj_pos_w + offset_w


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel.

    Uses dynamic z offset: starts high (0.15) to approach from above,
    then lowers to grasp position (0.08) as xy alignment improves.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    # Get EE position first for dynamic offset calculation
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]

    target_pos_w = _compute_grasp_target_pos_w(
        env,
        obj,
        ee_pos_w,
        use_dynamic_z=True,
        dynamic_z_state_attr="_reach_dynamic_z_prev_left",
    )
    _maybe_visualize_approach_target_all(env, target_pos_w, obj.data.root_quat_w, marker_attr="_debug_approach_target_left")

    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)
    reach_reward = 1 - torch.tanh(dist / std)

    # Suppress reaching reward when cup is pushed in XY during approach.
    cfg = getattr(env, "cfg", None)
    disp_free = float(getattr(cfg, "reach_displacement_free_threshold", 0.005))
    disp_scale = float(getattr(cfg, "reach_displacement_suppress_scale", 0.01))
    current_xy = obj.data.root_pos_w[:, :2]
    initial_xy = _get_episode_initial_object_xy(env, obj, "_cup_initial_xy_w_left")
    displacement_xy = torch.norm(current_xy - initial_xy, dim=1)
    displacement_excess = torch.clamp(displacement_xy - disp_free, min=0.0)
    if disp_scale > 0.0:
        reach_reward = reach_reward * torch.exp(-displacement_excess / disp_scale)

    reached_stable = _is_reaching_stably_complete(env, object_cfg, eef_link_name)
    return (1.0 - reached_stable) * reach_reward


def object_ee_distance_fine(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Fine-grained reaching reward toward the static grasp target (use_dynamic_z=False).

    Provides gradient to guide the EE from the dynamic target position
    all the way down to the actual grasp position (Z=offset[2]).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]

    target_pos_w = _compute_grasp_target_pos_w(
        env, obj, ee_pos_w, use_dynamic_z=False,
    )

    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)
    reach_reward = 1 - torch.tanh(dist / std)

    reached_stable = _is_reaching_stably_complete(env, object_cfg, eef_link_name)
    return (1.0 - reached_stable) * reach_reward


def _maybe_visualize_approach_target_all(
    env: ManagerBasedRLEnv,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    marker_attr: str,
) -> None:
    cfg = getattr(env, "cfg", None)
    if cfg is None or not getattr(cfg, "debug_approach_target_vis", True):
        return

    interval = int(getattr(cfg, "debug_approach_target_vis_interval", 10))
    step_count = int(getattr(env, "common_step_counter", 0))
    if interval > 1 and (step_count % interval) != 0:
        return

    if not hasattr(env, marker_attr):
        marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Debug/ApproachTargetLeft")
        marker_cfg.markers["frame"].scale = (0.04, 0.04, 0.04)
        marker = VisualizationMarkers(marker_cfg)
        marker.set_visibility(True)
        setattr(env, marker_attr, marker)

    marker = getattr(env, marker_attr)
    marker.visualize(target_pos_w, target_quat_w)


def _object_eef_any_axis_alignment(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Maximum absolute alignment between any EE axis and any object axis."""
    object_quat = env.scene[object_cfg.name].data.root_quat_w
    body_quat_w = env.scene["robot"].data.body_quat_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_quat = body_quat_w[:, eef_idx]

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)

    eef_axes = [quat_apply(eef_quat, x_axis), quat_apply(eef_quat, y_axis), quat_apply(eef_quat, z_axis)]
    obj_axes = [quat_apply(object_quat, x_axis), quat_apply(object_quat, y_axis), quat_apply(object_quat, z_axis)]

    max_align = torch.zeros(env.num_envs, device=env.device)
    for eef_axis in eef_axes:
        for obj_axis in obj_axes:
            align = torch.abs(torch.sum(eef_axis * obj_axis, dim=1))
            max_align = torch.maximum(max_align, align)
    return max_align


def eef_to_object_orientation(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Reward the end-effector aligning with any object axis (loose tanh-kernel)."""
    max_align = _object_eef_any_axis_alignment(env, eef_link_name, object_cfg)
    error = 1.0 - max_align
    return 1 - torch.tanh(error / std)


def eef_z_perpendicular_object_z(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str = "ll_dg_ee",
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Reward 90-degree alignment between EE +Z axis and object +Z axis."""
    object_quat = env.scene[object_cfg.name].data.root_quat_w
    body_quat_w = env.scene["robot"].data.body_quat_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_quat = body_quat_w[:, eef_idx]

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=object_quat.dtype).repeat(env.num_envs, 1)
    ee_z = quat_apply(eef_quat, z_axis)
    obj_z = quat_apply(object_quat, z_axis)

    cos_theta = torch.sum(ee_z * obj_z, dim=1).clamp(-1.0, 1.0)
    error = torch.abs(cos_theta)
    orientation_reward = 1 - torch.tanh(error / std)
    reached_stable = _is_reaching_stably_complete(env, object_cfg, eef_link_name)
    return (1.0 - reached_stable) * orientation_reward


def _is_reaching_complete(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
    reach_threshold: float = 0.01,
) -> torch.Tensor:
    """Check if EE has reached the grasp position (z_low offset near object).

    Returns a boolean mask (float 0/1) per environment.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    target_pos_w = _compute_grasp_target_pos_w(env, obj, ee_pos_w, use_dynamic_z=False)

    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)
    return (dist < reach_threshold).float()


def _reaching_soft_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Continuous [0, 1] gate based on EE distance to static grasp target."""
    cfg = getattr(env, "cfg", None)
    near = float(getattr(cfg, "reach_soft_gate_near", 0.02))
    far = float(getattr(cfg, "reach_soft_gate_far", 0.08))
    far = max(far, near + 1e-6)

    obj: RigidObject = env.scene[object_cfg.name]
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
    target_pos_w = _compute_grasp_target_pos_w(env, obj, ee_pos_w, use_dynamic_z=False)
    dist = torch.norm(target_pos_w - ee_pos_w, dim=1)

    gate = torch.clamp((far - dist) / (far - near), 0.0, 1.0)
    return gate


def _is_reaching_stably_complete(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Gate progression after reaching is maintained for multiple consecutive steps."""
    cfg = getattr(env, "cfg", None)
    reach_threshold = float(getattr(cfg, "reach_switch_threshold", 0.01))
    hold_steps = int(getattr(cfg, "reach_switch_hold_steps", 10))
    hold_steps = max(1, hold_steps)

    reached_now = _is_reaching_complete(env, object_cfg, eef_link_name, reach_threshold=reach_threshold)
    reached_now_i64 = reached_now.to(dtype=torch.int64)

    if not hasattr(env, "_reach_hold_counter_left"):
        env._reach_hold_counter_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
    counter = env._reach_hold_counter_left

    # Update once per sim step even if multiple reward terms query this gate.
    step_count = int(getattr(env, "common_step_counter", -1))
    if not hasattr(env, "_reach_hold_counter_left_last_step"):
        env._reach_hold_counter_left_last_step = -2
    if env._reach_hold_counter_left_last_step != step_count:
        # Reset counter at episode boundary.
        reset_mask = (env.episode_length_buf == 0).squeeze(-1)
        counter[reset_mask] = 0

        counter = torch.where(reached_now_i64 > 0, counter + 1, torch.zeros_like(counter))
        env._reach_hold_counter_left = counter
        env._reach_hold_counter_left_last_step = step_count

    return (counter >= hold_steps).to(dtype=reached_now.dtype)


def _reaching_progress_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link_name: str,
) -> torch.Tensor:
    """Combine soft and stable gates for robust phase transition."""
    stable = _is_reaching_stably_complete(env, object_cfg, eef_link_name)
    soft = _reaching_soft_gate(env, object_cfg, eef_link_name)
    soft_relaxed = torch.clamp(soft * 1.2, 0.0, 1.0)
    return torch.maximum(stable, soft_relaxed)


def _left_finger_contact_flags(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    contact_threshold: float = 0.02,
) -> torch.Tensor:
    """Estimate per-finger contact flags from contact sensor forces."""
    del object_cfg  # kept for backward signature compatibility
    contact_sensor = env.scene[sensor_cfg.name]
    force_magnitudes = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    sensor_body_names = getattr(contact_sensor, "body_names", None)
    if sensor_body_names is None:
        sensor_body_names = getattr(contact_sensor.data, "body_names", None)
    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]
        sensor_body_names = _select_sensor_body_names(sensor_body_names, sensor_cfg.body_ids)
    return _finger_contact_flags_from_sensor(force_magnitudes, contact_threshold, sensor_body_names)


def contact_finger_coverage_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
    contact_threshold: float = 0.02,
    min_fingers_bonus: int = 4,
    bonus_scale: float = 1.0,
) -> torch.Tensor:
    """Reward broader multi-finger coverage to avoid 2-3-finger local optima."""
    contact_flags = _left_finger_contact_flags(
        env,
        sensor_cfg=sensor_cfg,
        object_cfg=object_cfg,
        contact_threshold=contact_threshold,
    )
    num_fingers = contact_flags.sum(dim=1).float()
    coverage = num_fingers / 5.0

    min_fingers_bonus = max(1, min(5, int(min_fingers_bonus)))
    bonus_span = float(max(1, 6 - min_fingers_bonus))
    bonus = torch.clamp((num_fingers - float(min_fingers_bonus - 1)) / bonus_span, 0.0, 1.0)

    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * (coverage + float(bonus_scale) * bonus)


def strict_grasp_lift_success(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
    contact_threshold: float = 0.02,
    required_fingers: int = 4,
    minimal_height: float = 0.04,
    hold_steps: int = 8,
) -> torch.Tensor:
    """Binary success metric: multi-finger grasp maintained while object is lifted."""
    obj: RigidObject = env.scene[object_cfg.name]
    contact_flags = _left_finger_contact_flags(
        env,
        sensor_cfg=sensor_cfg,
        object_cfg=object_cfg,
        contact_threshold=contact_threshold,
    )
    num_fingers = contact_flags.sum(dim=1)
    required_fingers = max(1, min(5, int(required_fingers)))
    hold_steps = max(1, int(hold_steps))

    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    success_now = (num_fingers >= required_fingers) & (obj.data.root_pos_w[:, 2] > minimal_height) & (reached_gate > 0.2)

    if not hasattr(env, "_strict_grasp_success_counter"):
        env._strict_grasp_success_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
    counter = env._strict_grasp_success_counter

    # Update once per sim step even if queried by multiple terms.
    step_count = int(getattr(env, "common_step_counter", -1))
    if not hasattr(env, "_strict_grasp_success_last_step"):
        env._strict_grasp_success_last_step = -2
    if env._strict_grasp_success_last_step != step_count:
        reset_mask = (env.episode_length_buf == 0).squeeze(-1)
        counter[reset_mask] = 0
        counter = torch.where(success_now, counter + 1, torch.zeros_like(counter))
        env._strict_grasp_success_counter = counter
        env._strict_grasp_success_last_step = step_count

    return (counter >= hold_steps).to(dtype=obj.data.root_pos_w.dtype)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Binary reward if object is lifted above minimal height.

    Only activates when EE has reached the grasp position first.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    reached = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached * (obj.data.root_pos_w[:, 2] > minimal_height).float()


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward tracking the goal pose using tanh-kernel.

    Only activates when EE has reached the grasp position first.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    reached = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached * (obj.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_displacement_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    threshold: float = 0.02,
) -> torch.Tensor:
    """Penalize object movement from initial position (XY only)."""
    obj: RigidObject = env.scene[object_cfg.name]

    current_pos = obj.data.root_pos_w[:, :2]
    initial_pos = _get_episode_initial_object_xy(env, obj, "_cup_initial_xy_w_left")

    displacement = torch.norm(current_pos - initial_pos, dim=1)
    penalty = torch.clamp(displacement - threshold, min=0.0)

    return penalty


def finger_normal_range_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize left thumb+pinky joints going outside their normal curl range.

    Returns total violation amount (positive). Use negative weight in config.
    Joints outside the normal range (e.g. bending backward) accumulate violation in radians.
    """
    robot = env.scene["robot"]

    # Left hand normal ranges (from user-confirmed curl directions)
    # 1_1 (thumb spread) excluded - full range is acceptable
    _RANGES = {
        "lj_dg_1_2": (0.0, 1.571),      # positive = curl
        "lj_dg_1_3": (-1.571, 0.0),      # negative = curl
        "lj_dg_1_4": (-1.571, 0.0),      # negative = curl
        "lj_dg_5_1": (-0.1, 0.1),        # should stay near 0
        "lj_dg_5_2": (-0.611, 0.05),     # 0.0 ideal, slight positive tolerance
        "lj_dg_5_3": (0.0, 1.571),       # positive = curl
        "lj_dg_5_4": (0.0, 1.571),       # positive = curl
    }

    total_violation = torch.zeros(env.num_envs, device=env.device)

    for joint_name, (lo, hi) in _RANGES.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_violation += torch.clamp(lo - pos, min=0.0) + torch.clamp(pos - hi, min=0.0)

    return total_violation


def finger_reaching_pose_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward left thumb+pinky staying near initial open pose during reaching.

    Prevents excessive curling into the palm while approaching.
    Deactivates once reaching is stably complete to allow free grasping.
    """
    robot = env.scene["robot"]

    # Target = initial positions (open/ready pose)
    _TARGETS = {
        "lj_dg_1_2": 1.571,    # max open
        "lj_dg_1_3": 0.0,
        "lj_dg_1_4": 0.0,
        "lj_dg_5_3": 0.0,
        "lj_dg_5_4": 0.0,
    }

    total_sq_error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in _TARGETS.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_sq_error += (pos - target) ** 2

    reward = 1.0 - torch.tanh(total_sq_error / std)

    reached_stable = _is_reaching_stably_complete(env, object_cfg, eef_link_name)
    return (1.0 - reached_stable) * reward


def finger_grasp_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward all left fingers closing toward grasp pose after reaching is complete.

    Only active after _is_reaching_stably_complete.
    Acts like a binary gripper: maximally close all fingers.
    """
    robot = env.scene["robot"]

    _CLOSE_POSE = {
        # Thumb
        "lj_dg_1_1": 0.0, "lj_dg_1_2": 1.4, "lj_dg_1_3": -0.5, "lj_dg_1_4": -0.9,
        # Index
        "lj_dg_2_1": 0.0, "lj_dg_2_2": 0.5, "lj_dg_2_3": 0.8, "lj_dg_2_4": 1.0,
        # Middle
        "lj_dg_3_1": 0.0, "lj_dg_3_2": 0.5, "lj_dg_3_3": 0.8, "lj_dg_3_4": 1.0,
        # Ring
        "lj_dg_4_1": 0.0, "lj_dg_4_2": 0.5, "lj_dg_4_3": 0.8, "lj_dg_4_4": 1.0,
        # Pinky
        "lj_dg_5_1": 0.0, "lj_dg_5_2": 0.0, "lj_dg_5_3": 0.9, "lj_dg_5_4": 0.9,
    }

    total_sq_error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in _CLOSE_POSE.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_sq_error += (pos - target) ** 2

    reward = 1.0 - torch.tanh(total_sq_error / std)

    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * reward


def contact_persistence_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_contacts: int = 3,
    contact_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward maintaining sufficient finger-level contacts."""
    contact_sensor = env.scene[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w
    force_magnitudes = torch.norm(contact_forces, dim=-1)
    sensor_body_names = getattr(contact_sensor, "body_names", None)
    if sensor_body_names is None:
        sensor_body_names = getattr(contact_sensor.data, "body_names", None)
    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]
        sensor_body_names = _select_sensor_body_names(sensor_body_names, sensor_cfg.body_ids)
    finger_flags = _finger_contact_flags_from_sensor(force_magnitudes, contact_threshold, sensor_body_names)
    num_contacts = finger_flags.sum(dim=-1).float()
    reward = torch.clamp(num_contacts / float(max(min_contacts, 1)), 0.0, 1.0)
    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * reward


def slip_magnitude_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    max_slip: float = 0.15,
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 0.05,
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Penalty for fingertip-object relative slip."""
    robot = env.scene[robot_cfg.name]
    link_vel = robot.data.body_lin_vel_w
    if robot_cfg.body_ids is not None:
        link_vel = link_vel[:, robot_cfg.body_ids, :]

    obj = env.scene[object_cfg.name]
    obj_vel = obj.data.root_lin_vel_w.unsqueeze(1)
    rel_vel = link_vel - obj_vel
    slip_mag = torch.norm(rel_vel, dim=-1)
    avg_slip = slip_mag.mean(dim=-1)

    penalty = 1.0 - torch.exp(-torch.square(avg_slip / max(max_slip, 1e-6)))

    if sensor_cfg is not None:
        contact_sensor = env.scene[sensor_cfg.name]
        force_magnitudes = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
        if sensor_cfg.body_ids is not None:
            force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]
        has_contact = (force_magnitudes > contact_threshold).any(dim=-1)
        penalty = penalty * has_contact.float()

    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * penalty


def normal_force_stability_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward smooth contact-force changes over time."""
    contact_sensor = env.scene[sensor_cfg.name]
    force_magnitudes = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]

    buffer_name = f"_prev_force_mags_{sensor_cfg.name}"
    if hasattr(env, buffer_name):
        prev_forces = getattr(env, buffer_name)
        delta = torch.abs(force_magnitudes - prev_forces)
        stability = 1.0 / (1.0 + delta.mean(dim=-1))
    else:
        stability = torch.ones(env.num_envs, device=env.device)
    setattr(env, buffer_name, force_magnitudes.clone())

    has_contact = (force_magnitudes > contact_threshold).any(dim=-1)
    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * stability * has_contact.float()


def force_spike_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    spike_threshold: float = 10.0,
    contact_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Penalty for abrupt contact-force spikes."""
    contact_sensor = env.scene[sensor_cfg.name]
    force_magnitudes = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]

    buffer_name = f"_prev_force_rate_{sensor_cfg.name}"
    if hasattr(env, buffer_name):
        prev_forces = getattr(env, buffer_name)
        force_rate = torch.abs(force_magnitudes - prev_forces) / max(env.step_dt, 1e-6)
        max_rate = force_rate.max(dim=-1)[0]
        penalty = torch.clamp((max_rate - spike_threshold) / max(spike_threshold, 1e-6), 0.0, 1.0)
    else:
        penalty = torch.zeros(env.num_envs, device=env.device)
    setattr(env, buffer_name, force_magnitudes.clone())

    has_contact = (force_magnitudes > contact_threshold).any(dim=-1)
    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * penalty * has_contact.float()


def overgrip_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target_force_range: tuple[float, float] = (1.0, 12.0),
    contact_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Penalty for under/over target grip force band."""
    contact_sensor = env.scene[sensor_cfg.name]
    force_magnitudes = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]

    total_force = force_magnitudes.sum(dim=-1)
    min_force, max_force = target_force_range

    undergrip = torch.clamp(min_force - total_force, 0.0, max(min_force, 1e-6)) / max(min_force, 1e-6)
    overgrip = torch.clamp(total_force - max_force, 0.0, max(max_force, 1e-6)) / max(max_force, 1e-6)
    penalty = undergrip + overgrip

    has_contact = (force_magnitudes > contact_threshold).any(dim=-1)
    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * penalty * has_contact.float()
