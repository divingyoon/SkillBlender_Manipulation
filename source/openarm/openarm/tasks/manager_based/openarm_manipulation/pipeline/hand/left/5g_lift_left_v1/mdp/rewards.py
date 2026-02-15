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

    # Keep reaching reward even after reaching complete (no sudden drop to 0)
    # This prevents policy from avoiding threshold boundary
    return reach_reward


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

    # Keep reaching reward even after reaching complete
    return reach_reward


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

    result = (counter >= hold_steps).to(dtype=reached_now.dtype)

    # Debug output (every 50 steps, env 0 only)
    debug_enabled = getattr(cfg, "debug_reaching", True)
    if debug_enabled and step_count % 50 == 0:
        obj = env.scene[object_cfg.name]
        eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
        ee_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx]
        target_pos_w = _compute_grasp_target_pos_w(env, obj, ee_pos_w, use_dynamic_z=False)
        dist = torch.norm(target_pos_w - ee_pos_w, dim=1)

        print(f"[Step {step_count}] EE-Target dist: {dist[0].item():.4f}m | "
              f"Threshold: {reach_threshold:.3f}m | "
              f"Reached: {reached_now[0].item():.0f} | "
              f"Counter: {counter[0].item()}/{hold_steps} | "
              f"Grasp Active: {result[0].item():.0f}")

    return result


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
    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * (obj.data.root_pos_w[:, 2] > minimal_height).float()


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
    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * (obj.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


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


def thumb_reaching_pose_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward thumb (finger 1) staying near initial open pose during reaching.

    Deactivates once reaching is stably complete to allow free grasping.
    """
    robot = env.scene["robot"]

    # Thumb target = open pose
    _TARGETS = {
        "lj_dg_1_2": 1.571,    # max open
        "lj_dg_1_3": 0.0,
        "lj_dg_1_4": 0.0,
    }

    total_sq_error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in _TARGETS.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_sq_error += (pos - target) ** 2

    reward = 1.0 - torch.tanh(total_sq_error / std)

    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return (1.0 - reached_gate) * reward


def pinky_reaching_pose_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward pinky (finger 5) staying near initial open pose during reaching.

    Deactivates once reaching is stably complete to allow free grasping.
    """
    robot = env.scene["robot"]

    # Pinky target = open pose
    _TARGETS = {
        "lj_dg_5_3": 0.0,
        "lj_dg_5_4": 0.0,
    }

    total_sq_error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in _TARGETS.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_sq_error += (pos - target) ** 2

    reward = 1.0 - torch.tanh(total_sq_error / std)

    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return (1.0 - reached_gate) * reward


def finger_contact_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Reward for fingers making contact with the object.

    Uses contact force sensor to detect physical contact between
    finger links and the cup. Only active after reaching is complete.
    """
    # Finger body indices to check for contact
    _FINGER_BODIES = [
        "ll_dg_1_4", "ll_dg_2_4", "ll_dg_3_4", "ll_dg_4_4", "ll_dg_5_4",  # fingertips
        "ll_dg_1_3", "ll_dg_2_3", "ll_dg_3_3", "ll_dg_4_3", "ll_dg_5_3",  # middle phalanges
    ]

    robot = env.scene["robot"]
    obj = env.scene[object_cfg.name]

    # Get fingertip positions
    finger_pos_list = []
    for body_name in _FINGER_BODIES:
        if body_name in robot.data.body_names:
            body_idx = robot.data.body_names.index(body_name)
            finger_pos_list.append(robot.data.body_pos_w[:, body_idx])

    if not finger_pos_list:
        return torch.zeros(env.num_envs, device=env.device)

    finger_positions = torch.stack(finger_pos_list, dim=1)  # (num_envs, num_fingers, 3)
    obj_pos = obj.data.root_pos_w.unsqueeze(1)  # (num_envs, 1, 3)

    # Distance from each finger to object center
    distances = torch.norm(finger_positions - obj_pos, dim=2)  # (num_envs, num_fingers)

    # Count fingers within contact threshold (e.g., 5cm from cup center)
    contact_threshold = 0.06
    contacts = (distances < contact_threshold).float()
    num_contacts = contacts.sum(dim=1)  # (num_envs,)

    # Normalize by max possible contacts
    max_contacts = float(len(finger_pos_list))
    contact_ratio = num_contacts / max_contacts

    # Only reward after reaching is complete
    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * contact_ratio


def thumb_grasp_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward thumb (finger 1) for closing - velocity + position based.

    Combines:
    - Velocity reward: moving in closing direction
    - Position reward: being in closed position (even if not moving)

    Only active after _is_reaching_stably_complete.
    """
    robot = env.scene["robot"]

    # Thumb targets (closed position)
    _CLOSE_TARGETS = {
        "lj_dg_1_2": 2.5,   # increases to curl (limit: 3.14)
        "lj_dg_1_3": -1.4,  # decreases to curl (limit: -1.57)
        "lj_dg_1_4": -1.4,  # decreases to curl (limit: -1.57)
    }

    # Curl directions for velocity
    _CURL_DIRECTIONS = {
        "lj_dg_1_2": +1.0,
        "lj_dg_1_3": -1.0,
        "lj_dg_1_4": -1.0,
    }

    # Velocity-based reward
    curl_velocity_sum = torch.zeros(env.num_envs, device=env.device)
    for joint_name, direction in _CURL_DIRECTIONS.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        vel = robot.data.joint_vel[:, joint_idx]
        curl_velocity_sum += direction * vel

    velocity_reward = torch.tanh(curl_velocity_sum / std)
    velocity_reward = torch.clamp(velocity_reward, min=0.0)

    # Position-based reward (how close to closed position)
    total_sq_error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in _CLOSE_TARGETS.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_sq_error += (pos - target) ** 2

    position_reward = 1.0 - torch.tanh(total_sq_error / std)

    # Combine: velocity (0.3) + position (0.7)
    reward = 0.3 * velocity_reward + 0.7 * position_reward

    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * reward


def pinky_grasp_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward pinky (finger 5) for closing - velocity + position based.

    Combines:
    - Velocity reward: moving in closing direction
    - Position reward: being in closed position (even if not moving)

    Only active after _is_reaching_stably_complete.
    """
    robot = env.scene["robot"]

    # Pinky targets (closed position)
    _CLOSE_TARGETS = {
        "lj_dg_5_3": 1.5,  # increases to curl (limit: 1.57)
        "lj_dg_5_4": 1.5,  # increases to curl (limit: 1.57)
    }

    # Curl directions for velocity
    _CURL_DIRECTIONS = {
        "lj_dg_5_3": +1.0,
        "lj_dg_5_4": +1.0,
    }

    # Velocity-based reward
    curl_velocity_sum = torch.zeros(env.num_envs, device=env.device)
    for joint_name, direction in _CURL_DIRECTIONS.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        vel = robot.data.joint_vel[:, joint_idx]
        curl_velocity_sum += direction * vel

    velocity_reward = torch.tanh(curl_velocity_sum / std)
    velocity_reward = torch.clamp(velocity_reward, min=0.0)

    # Position-based reward (how close to closed position)
    total_sq_error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in _CLOSE_TARGETS.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_sq_error += (pos - target) ** 2

    position_reward = 1.0 - torch.tanh(total_sq_error / std)

    # Combine: velocity (0.3) + position (0.7)
    reward = 0.3 * velocity_reward + 0.7 * position_reward

    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)
    return reached_gate * reward


def synergy_grip_reward(
    env: ManagerBasedRLEnv,
    action_name: str = "left_hand_action",
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    eef_link_name: str = "ll_dg_ee",
) -> torch.Tensor:
    """Reward synergy fingers (2,3,4) based on reaching phase.

    - Before reaching complete: small reward for OPEN (grip_strength < 0)
    - After reaching complete: full reward for CLOSED (grip_strength > 0)
      Plus position-based reward for actual joint positions being closed.

    grip_strength in [-1, 1]: -1 = open, +1 = closed
    """
    robot = env.scene["robot"]

    # Get the raw action (grip_strength) from action manager
    action_term = env.action_manager.get_term(action_name)
    grip_strength = action_term.raw_actions[:, 0]  # shape: (num_envs,)

    reached_gate = _reaching_progress_gate(env, object_cfg, eef_link_name)

    # Before reaching: small reward for open (0.2 scale to reduce incentive to stay open)
    open_reward = torch.clamp((1.0 - grip_strength) / 2.0, min=0.0, max=1.0) * 0.2

    # After reaching: reward based on action command + actual joint positions
    action_reward = torch.clamp((grip_strength + 1.0) / 2.0, min=0.0, max=1.0)

    # Position-based reward for synergy fingers (2,3,4)
    _CLOSE_TARGETS = {
        "lj_dg_2_2": 1.9, "lj_dg_2_3": 1.5, "lj_dg_2_4": 1.5,
        "lj_dg_3_2": 1.85, "lj_dg_3_3": 1.5, "lj_dg_3_4": 1.5,
        "lj_dg_4_2": 1.8, "lj_dg_4_3": 1.5, "lj_dg_4_4": 1.5,
    }

    total_sq_error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target in _CLOSE_TARGETS.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        pos = robot.data.joint_pos[:, joint_idx]
        total_sq_error += (pos - target) ** 2

    position_reward = 1.0 - torch.tanh(total_sq_error / 5.0)  # std=5.0 for 9 joints

    # Combine action (0.3) + position (0.7) for close reward
    close_reward = 0.3 * action_reward + 0.7 * position_reward

    # Combine based on reaching state
    reward = (1.0 - reached_gate) * open_reward + reached_gate * close_reward

    return reward
