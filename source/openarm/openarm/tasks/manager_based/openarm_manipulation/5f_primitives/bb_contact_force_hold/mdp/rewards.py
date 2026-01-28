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

"""Reward functions for Primitive B: Contact Force Hold.

Task-agnostic grasp maintenance rewards. NO GWS/epsilon or task success metrics.
"""

from __future__ import annotations

import torch

DEBUG_REWARD_LOGS = True
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

from ...common.robot_cfg import LEFT_CONTACT_LINKS, RIGHT_CONTACT_LINKS


__all__ = [
    "contact_persistence_reward",
    "contact_persistence_reward_multi",
    "missing_contact_penalty",
    "slip_magnitude_penalty",
    "stable_action_rate_l2",
    "force_spike_penalty",
    "force_spike_penalty_multi",
    "overgrip_penalty",
    "overgrip_penalty_multi",
    "grip_force_in_band",
    "grip_force_in_band_multi",
    "pushout_velocity_penalty",
]


def _contact_force_magnitudes(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
) -> torch.Tensor:
    """Collect contact force magnitudes for multiple sensors.

    Returns:
        Force magnitudes (num_envs, num_sensors)
    """
    def _sensor_to_link(sensor_name: str) -> str | None:
        if sensor_name == "left_contact_sensor":
            return LEFT_CONTACT_LINKS[0]
        if sensor_name.startswith("left_contact_sensor_"):
            idx = int(sensor_name.split("_")[-1]) - 1
            if 0 <= idx < len(LEFT_CONTACT_LINKS):
                return LEFT_CONTACT_LINKS[idx]
        if sensor_name == "right_contact_sensor":
            return RIGHT_CONTACT_LINKS[0]
        if sensor_name.startswith("right_contact_sensor_"):
            idx = int(sensor_name.split("_")[-1]) - 1
            if 0 <= idx < len(RIGHT_CONTACT_LINKS):
                return RIGHT_CONTACT_LINKS[idx]
        return None

    robot = env.scene["robot"]
    mags = []
    none_sensors: list[str] = []
    for sensor_name in sensor_names:
        contact_sensor: ContactSensor = env.scene[sensor_name]
        link_name = _sensor_to_link(sensor_name)
        link_quat_w = None
        if link_name is not None and link_name in robot.body_names:
            body_id = robot.body_names.index(link_name)
            link_quat_w = robot.data.body_quat_w[:, body_id]
        # Prefer filtered contact forces if available to avoid self-collision leakage.
        if getattr(contact_sensor.data, "force_matrix_w", None) is not None:
            # (num_envs, num_bodies, num_filters, 3) -> (num_envs, num_bodies)
            force_matrix_w = contact_sensor.data.force_matrix_w
            if link_quat_w is not None:
                force_flat = force_matrix_w.reshape(force_matrix_w.shape[0], -1, 3)
                quat_flat = link_quat_w.unsqueeze(1).expand(-1, force_flat.shape[1], -1)
                force_flat = math_utils.quat_apply_inverse(quat_flat, force_flat)
                force_matrix_w = force_flat.reshape_as(force_matrix_w)
            force_magnitudes = torch.norm(force_matrix_w, dim=-1).max(dim=-1)[0]
        else:
            none_sensors.append(sensor_name)
            contact_forces = contact_sensor.data.net_forces_w
            if link_quat_w is not None and contact_forces.dim() >= 2:
                force_flat = contact_forces.reshape(contact_forces.shape[0], -1, 3)
                quat_flat = link_quat_w.unsqueeze(1).expand(-1, force_flat.shape[1], -1)
                force_flat = math_utils.quat_apply_inverse(quat_flat, force_flat)
                contact_forces = force_flat.reshape_as(contact_forces)
            force_magnitudes = torch.norm(contact_forces, dim=-1)
        if force_magnitudes.dim() == 2:
            force_magnitudes = force_magnitudes.max(dim=-1)[0]
        mags.append(force_magnitudes)
    # Debug: report which sensors lack force_matrix_w (throttled).
    step = int(getattr(env, "common_step_counter", 0))
    if DEBUG_REWARD_LOGS and step % 200 == 0 and getattr(env, "_debug_fmw_last", -1) != step:
        if none_sensors:
            print(f"[fmw_debug] step={step} force_matrix_w=None sensors={none_sensors}")
        else:
            print(f"[fmw_debug] step={step} all sensors have force_matrix_w")
        env._debug_fmw_last = step
    return torch.stack(mags, dim=-1)


def _contact_force_magnitudes_unfiltered(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
) -> torch.Tensor:
    """Collect unfiltered contact force magnitudes (uses net_forces_w only)."""
    def _sensor_to_link(sensor_name: str) -> str | None:
        if sensor_name == "left_contact_sensor":
            return LEFT_CONTACT_LINKS[0]
        if sensor_name.startswith("left_contact_sensor_"):
            idx = int(sensor_name.split("_")[-1]) - 1
            if 0 <= idx < len(LEFT_CONTACT_LINKS):
                return LEFT_CONTACT_LINKS[idx]
        if sensor_name == "right_contact_sensor":
            return RIGHT_CONTACT_LINKS[0]
        if sensor_name.startswith("right_contact_sensor_"):
            idx = int(sensor_name.split("_")[-1]) - 1
            if 0 <= idx < len(RIGHT_CONTACT_LINKS):
                return RIGHT_CONTACT_LINKS[idx]
        return None

    robot = env.scene["robot"]
    mags = []
    for sensor_name in sensor_names:
        contact_sensor: ContactSensor = env.scene[sensor_name]
        link_name = _sensor_to_link(sensor_name)
        link_quat_w = None
        if link_name is not None and link_name in robot.body_names:
            body_id = robot.body_names.index(link_name)
            link_quat_w = robot.data.body_quat_w[:, body_id]

        contact_forces = contact_sensor.data.net_forces_w
        if link_quat_w is not None and contact_forces.dim() >= 2:
            force_flat = contact_forces.reshape(contact_forces.shape[0], -1, 3)
            quat_flat = link_quat_w.unsqueeze(1).expand(-1, force_flat.shape[1], -1)
            force_flat = math_utils.quat_apply_inverse(quat_flat, force_flat)
            contact_forces = force_flat.reshape_as(contact_forces)
        force_magnitudes = torch.norm(contact_forces, dim=-1)
        if force_magnitudes.dim() == 2:
            force_magnitudes = force_magnitudes.max(dim=-1)[0]
        mags.append(force_magnitudes)
    return torch.stack(mags, dim=-1)


def _contact_force_magnitudes_filtered(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
) -> torch.Tensor:
    """Collect filtered contact force magnitudes (sensor-body vs filtered prims only).

    If a sensor has no filtered force matrix, returns zeros for that sensor.
    """
    mags = []
    for sensor_name in sensor_names:
        contact_sensor: ContactSensor = env.scene[sensor_name]
        force_matrix_w = getattr(contact_sensor.data, "force_matrix_w", None)
        if force_matrix_w is None:
            mags.append(torch.zeros(env.num_envs, device=env.device))
            continue
        force_magnitudes = torch.norm(force_matrix_w, dim=-1).max(dim=-1)[0]
        if force_magnitudes.dim() == 2:
            force_magnitudes = force_magnitudes.max(dim=-1)[0]
        mags.append(force_magnitudes)
    return torch.stack(mags, dim=-1)


def missing_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    contact_threshold: float = 0.05,
    use_filtered: bool = False,
) -> torch.Tensor:
    """Penalty when none of the specified sensors are in contact.

    Returns 1.0 if all sensors are below threshold, else 0.0.
    """
    if use_filtered:
        force_magnitudes = _contact_force_magnitudes_filtered(env, sensor_names)
    else:
        force_magnitudes = _contact_force_magnitudes_unfiltered(env, sensor_names)
    contact_mask = force_magnitudes > contact_threshold
    any_contact = contact_mask.any(dim=-1)
    return (~any_contact).float()


def contact_persistence_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_contacts: int = 3,
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward for maintaining minimum number of contacts.

    Args:
        env: Environment instance
        sensor_cfg: Contact sensor configuration
        min_contacts: Minimum contacts for full reward
        contact_threshold: Force threshold for contact detection

    Returns:
        Reward in [0, 1] (num_envs,)
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # Get contact forces
    contact_forces = contact_sensor.data.net_forces_w
    force_magnitudes = torch.norm(contact_forces, dim=-1)

    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]

    # Count contacts above threshold
    num_contacts = (force_magnitudes > contact_threshold).sum(dim=-1).float()

    # Normalize by required contacts
    reward = torch.clamp(num_contacts / min_contacts, 0.0, 1.0)

    return reward


def contact_persistence_reward_multi(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    min_contacts: int = 3,
    contact_threshold: float = 0.1,
    use_filtered: bool = False,
) -> torch.Tensor:
    """Reward for maintaining minimum number of contacts (multi-sensor).

    Args:
        env: Environment instance
        sensor_names: List of contact sensor names
        min_contacts: Minimum contacts for full reward
        contact_threshold: Force threshold for contact detection

    Returns:
        Reward in [0, 1] (num_envs,)
    """
    if use_filtered:
        force_magnitudes = _contact_force_magnitudes_filtered(env, sensor_names)
    else:
        force_magnitudes = _contact_force_magnitudes(env, sensor_names)
    num_contacts = (force_magnitudes > contact_threshold).sum(dim=-1).float()

    # Debug: track per-episode success steps for left_contact_persistence variants.
    is_left_lj1 = (
        len(sensor_names) == 2
        and sensor_names[0] == "left_contact_sensor"
        and sensor_names[1] == "left_contact_sensor_2"
    )
    is_left_main = sensor_names == [
        "left_contact_sensor_3",
        "left_contact_sensor_4",
        "left_contact_sensor_5",
        "left_contact_sensor_6",
        "left_contact_sensor_7",
        "left_contact_sensor_8",
        "left_contact_sensor_9",
        "left_contact_sensor_10",
    ]
    if is_left_lj1 or is_left_main:
        # One-time debug: print sensor prims captured by PhysX (helps detect wrong prim_path).
        if DEBUG_REWARD_LOGS and not hasattr(env, "_debug_lcp_sensor_info"):
            try:
                sample_name = sensor_names[0]
                cs = env.scene[sample_name]
                print(
                    f"[lcp_sensor_info] name={sample_name} prim_path={cs.cfg.prim_path} "
                    f"num_bodies={cs.num_bodies} body_names={cs.body_names}"
                )
            except Exception as exc:
                print(f"[lcp_sensor_info] failed to read sensor info: {exc}")
            env._debug_lcp_sensor_info = True
        # Throttled debug: show max force + num_contacts for env 0.
        step = int(getattr(env, "common_step_counter", 0))
        if DEBUG_REWARD_LOGS and step % 200 == 0 and getattr(env, "_debug_lcp_force_last", -1) != step:
            max_force = force_magnitudes.max(dim=-1)[0]
            env0 = 0
            print(
                f"[lcp_force_debug] step={step} "
                f"term={'lj1' if is_left_lj1 else 'main'} "
                f"max_force={max_force[env0].item():.6f} "
                f"num_contacts={num_contacts[env0].item():.0f} "
                f"threshold={contact_threshold}"
            )
            env._debug_lcp_force_last = step
        if not hasattr(env, "_debug_lcp_counts"):
            env._debug_lcp_counts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
            env._debug_lcp_lj1_counts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        # Count success only when the per-term min_contacts is met.
        success_mask = (num_contacts >= float(min_contacts)).long()
        if is_left_main:
            env._debug_lcp_counts += success_mask
        if is_left_lj1:
            env._debug_lcp_lj1_counts += success_mask

        done_mask = env.termination_manager.terminated | env.termination_manager.time_outs
        if done_mask.any() and is_left_main:
            done_ids = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
            for env_id in done_ids.tolist():
                steps = int(env.episode_length_buf[env_id].item())
                lcp_steps = int(env._debug_lcp_counts[env_id].item())
                lj1_steps = int(env._debug_lcp_lj1_counts[env_id].item())
                if DEBUG_REWARD_LOGS:
                    print(
                        f"[lcp_debug] env={env_id} steps={steps} "
                        f"left_contact_persistence={lcp_steps} "
                        f"left_contact_persistence_lj_dg_1_4={lj1_steps}"
                    )
            env._debug_lcp_counts[done_mask] = 0
            env._debug_lcp_lj1_counts[done_mask] = 0

    # Temporary debug cache for left-hand contact counts.
    if any(name.startswith("left_contact_sensor") for name in sensor_names):
        frame_id = getattr(env, "_debug_reward_frame_id", 0) + 1
        setattr(env, "_debug_reward_frame_id", frame_id)
        setattr(env, "_debug_reward_last_contact_count", num_contacts.detach())

        cache = getattr(env, "_debug_reward_cache", {})
        cache["frame_id"] = frame_id
        cache.pop("slip_penalty", None)
        cache.pop("force_spike", None)
        cache.pop("overgrip", None)
        cache["slip_active"] = None
        cache["force_active"] = None
        cache["overgrip_active"] = None
        setattr(env, "_debug_reward_cache", cache)

    return torch.clamp(num_contacts / min_contacts, 0.0, 1.0)


def slip_magnitude_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = None,
    max_slip: float = 0.05,
    contact_sensor_names: list[str] | None = None,
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalty for tangential slip at contact points.

    Args:
        env: Environment instance
        robot_cfg: Robot config with body_names for contact links
        object_cfg: Optional object config for relative velocity
        max_slip: Maximum acceptable slip velocity (m/s)

    Returns:
        Normalized slip penalty (num_envs,)
    """
    robot = env.scene[robot_cfg.name]

    # Get contact link velocities
    link_vel = robot.data.body_lin_vel_w

    if robot_cfg.body_ids is not None:
        link_vel = link_vel[:, robot_cfg.body_ids, :]

    if object_cfg is not None:
        obj = env.scene[object_cfg.name]
        obj_vel = obj.data.root_lin_vel_w.unsqueeze(1)
        relative_vel = link_vel - obj_vel
    else:
        relative_vel = link_vel

    # Compute slip magnitude per link
    slip_mag = torch.norm(relative_vel, dim=-1)

    contact_count = None
    if contact_sensor_names is not None:
        contact_mags = _contact_force_magnitudes(env, contact_sensor_names)
        contact_mask = contact_mags > contact_threshold
        if slip_mag.shape[-1] != contact_mask.shape[-1]:
            min_count = min(slip_mag.shape[-1], contact_mask.shape[-1])
            slip_mag = slip_mag[:, :min_count]
            contact_mask = contact_mask[:, :min_count]
        contact_count = contact_mask.sum(dim=-1)
        slip_mag = slip_mag * contact_mask
        avg_slip = slip_mag.sum(dim=-1) / torch.clamp(contact_count, min=1)
    else:
        avg_slip = slip_mag.mean(dim=-1)

    # Smooth normalized penalty to avoid saturation.
    penalty = 1.0 - torch.exp(-torch.square(avg_slip / max_slip))
    if contact_count is not None:
        penalty = penalty * (contact_count > 0).float()

    # Temporary debug cache for left-hand slip.
    cache = getattr(env, "_debug_reward_cache", {})
    frame_id = getattr(env, "_debug_reward_frame_id", None)
    if frame_id is not None and cache.get("slip_frame_id") != frame_id:
        # First slip term this step corresponds to left hand (configured before right).
        cache["slip_penalty"] = penalty.detach()
        cache["slip_active"] = True
        cache["slip_frame_id"] = frame_id
        setattr(env, "_debug_reward_cache", cache)

    return penalty


def pushout_velocity_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    palm_link_name: str,
    outward_axis: str = "x",
    contact_sensor_names: list[str] | None = None,
    contact_threshold: float = 0.05,
    min_contacts: int = 1,
    eps: float = 0.03,
    smooth_scale: float = 0.1,
) -> torch.Tensor:
    """Penalty for object moving outward relative to the palm frame.

    Args:
        object_cfg: Object entity.
        robot_cfg: Robot entity for palm link lookup.
        palm_link_name: Link name used as palm frame.
        outward_axis: Axis in palm frame to treat as outward (+x/+y/+z).
        contact_sensor_names: Contact sensors to gate the penalty.
        contact_threshold: Force threshold for contact detection.
        min_contacts: Minimum contacts required to activate penalty.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    try:
        palm_id = robot.body_names.index(palm_link_name)
    except ValueError:
        return torch.zeros(env.num_envs, device=env.device)

    v_obj = obj.data.root_lin_vel_w
    v_palm = robot.data.body_lin_vel_w[:, palm_id]
    v_rel = v_obj - v_palm

    q_palm = robot.data.body_quat_w[:, palm_id]
    v_rel_palm = math_utils.quat_apply_inverse(q_palm, v_rel)

    axis = outward_axis.lower()
    if axis == "y":
        n_out = torch.tensor([0.0, 1.0, 0.0], device=env.device)
    elif axis == "z":
        n_out = torch.tensor([0.0, 0.0, 1.0], device=env.device)
    else:
        n_out = torch.tensor([1.0, 0.0, 0.0], device=env.device)

    v_out = (v_rel_palm * n_out).sum(dim=-1)
    v_eff = torch.clamp(v_out - eps, min=0.0)
    penalty = 1.0 - torch.exp(-((v_eff / smooth_scale) ** 2))

    if contact_sensor_names is not None and len(contact_sensor_names) > 0:
        contact_mags = _contact_force_magnitudes(env, contact_sensor_names)
        contact_count = (contact_mags > contact_threshold).sum(dim=-1)
        penalty = penalty * (contact_count >= min_contacts).float()

    return penalty


def stable_action_rate_l2(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    min_contacts: int = 3,
    contact_threshold: float = 0.05,
    slip_threshold: float = 0.02,
    spike_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize action rate only when grasp is stable."""
    force_magnitudes = _contact_force_magnitudes(env, sensor_names)
    contact_count = (force_magnitudes > contact_threshold).sum(dim=-1)

    # Slip condition
    robot = env.scene[robot_cfg.name]
    link_vel = robot.data.body_lin_vel_w
    if robot_cfg.body_ids is not None:
        link_vel = link_vel[:, robot_cfg.body_ids, :]
    obj = env.scene[object_cfg.name]
    obj_vel = obj.data.root_lin_vel_w.unsqueeze(1)
    slip_mag = torch.norm(link_vel - obj_vel, dim=-1).max(dim=-1)[0]

    # Force spike condition
    buffer_name = f"_prev_forces_stable_action_{'_'.join(sensor_names)}"
    if hasattr(env, buffer_name):
        prev_forces = getattr(env, buffer_name)
        dt = env.step_dt
        force_rate = torch.abs(force_magnitudes - prev_forces) / dt
        max_rate = force_rate.max(dim=-1)[0]
    else:
        max_rate = torch.zeros(env.num_envs, device=env.device)
    setattr(env, buffer_name, force_magnitudes.clone())

    stable_mask = (
        (contact_count >= min_contacts)
        & (slip_mag <= slip_threshold)
        & (max_rate <= spike_threshold)
    )
    action_rate = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return action_rate * stable_mask.float()


def force_spike_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    spike_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalty for sudden force spikes (impacts).

    Args:
        env: Environment instance
        sensor_cfg: Contact sensor configuration
        spike_threshold: Force rate threshold (N/s)

    Returns:
        Spike penalty (num_envs,)
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # Get current forces
    contact_forces = contact_sensor.data.net_forces_w
    force_magnitudes = torch.norm(contact_forces, dim=-1)

    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]

    # Store/retrieve previous forces
    buffer_name = f"_prev_forces_{sensor_cfg.name}"
    if hasattr(env, buffer_name):
        prev_forces = getattr(env, buffer_name)
        # Compute force rate of change
        dt = env.step_dt
        force_rate = torch.abs(force_magnitudes - prev_forces) / dt
        max_rate = force_rate.max(dim=-1)[0]
        penalty = torch.clamp((max_rate - spike_threshold) / spike_threshold, 0.0, 1.0)
    else:
        penalty = torch.zeros(env.num_envs, device=env.device)

    # Update buffer
    setattr(env, buffer_name, force_magnitudes.clone())

    return penalty


def force_spike_penalty_multi(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    spike_threshold: float = 10.0,
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalty for sudden force spikes across multiple sensors."""
    force_magnitudes = _contact_force_magnitudes(env, sensor_names)
    contact_count = (force_magnitudes > contact_threshold).sum(dim=-1)
    buffer_name = f"_prev_forces_{'_'.join(sensor_names)}"
    if hasattr(env, buffer_name):
        prev_forces = getattr(env, buffer_name)
        dt = env.step_dt
        force_rate = torch.abs(force_magnitudes - prev_forces) / dt
        max_rate = force_rate.max(dim=-1)[0]
        penalty = torch.clamp((max_rate - spike_threshold) / spike_threshold, 0.0, 1.0)
    else:
        penalty = torch.zeros(env.num_envs, device=env.device)
    setattr(env, buffer_name, force_magnitudes.clone())
    penalty = penalty * (contact_count > 0).float()

    # Temporary debug cache for left-hand force spikes.
    if any(name.startswith("left_contact_sensor") for name in sensor_names):
        cache = getattr(env, "_debug_reward_cache", {})
        cache["force_spike"] = penalty.detach()
        cache["force_active"] = True
        setattr(env, "_debug_reward_cache", cache)

    return penalty


def overgrip_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_force: float = 15.0,
) -> torch.Tensor:
    """Penalty for excessive grip force.

    Args:
        env: Environment instance
        sensor_cfg: Contact sensor configuration
        max_force: Maximum acceptable total force (N)

    Returns:
        Overgrip penalty (num_envs,)
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # Get force magnitudes
    contact_forces = contact_sensor.data.net_forces_w
    force_magnitudes = torch.norm(contact_forces, dim=-1)

    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]

    # Total force across all contacts
    total_force = force_magnitudes.sum(dim=-1)

    # Penalty for exceeding max
    excess = torch.clamp(total_force - max_force, 0.0, max_force)
    penalty = excess / max_force

    return penalty


def overgrip_penalty_multi(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    max_force: float = 15.0,
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalty for excessive grip force across multiple sensors."""
    force_magnitudes = _contact_force_magnitudes(env, sensor_names)
    contact_count = (force_magnitudes > contact_threshold).sum(dim=-1)
    total_force = force_magnitudes.sum(dim=-1)
    excess = torch.clamp(total_force - max_force, 0.0, max_force)
    penalty = excess / max_force
    penalty = penalty * (contact_count > 0).float()

    # Temporary debug cache for left-hand overgrip.
    if any(name.startswith("left_contact_sensor") for name in sensor_names):
        cache = getattr(env, "_debug_reward_cache", {})
        cache["overgrip"] = penalty.detach()
        cache["overgrip_active"] = True
        setattr(env, "_debug_reward_cache", cache)

        debug_max = getattr(env, "_debug_reward_max_steps", -1)
        debug_step = getattr(env, "_debug_reward_step", 0)
        if debug_max < 0 or debug_step < debug_max:
            contact_counts = getattr(env, "_debug_reward_last_contact_count", None)
            slip = cache.get("slip_penalty")
            force = cache.get("force_spike")
            overgrip = cache.get("overgrip")
            slip_active = cache.get("slip_active", None)
            force_active = cache.get("force_active", None)
            overgrip_active = cache.get("overgrip_active", None)

            if contact_counts is not None:
                contact_zero_pct = (contact_counts == 0).float().mean().item() * 100.0
                contact_mean = contact_counts.mean().item()
            else:
                contact_zero_pct = float("nan")
                contact_mean = float("nan")

            slip_mean = slip.mean().item() if slip is not None else float("nan")
            force_mean = force.mean().item() if force is not None else float("nan")
            overgrip_mean = overgrip.mean().item() if overgrip is not None else float("nan")

            slip_c0 = slip_c1 = float("nan")
            if contact_counts is not None and slip is not None:
                mask0 = contact_counts == 0
                mask1 = ~mask0
                if mask0.any():
                    slip_c0 = slip[mask0].mean().item()
                if mask1.any():
                    slip_c1 = slip[mask1].mean().item()

            b1_total = getattr(env, "_debug_b1_total", 0)
            b2_total = getattr(env, "_debug_b2_total", 0)
            b1_ratio = float("nan")
            if (b1_total + b2_total) > 0:
                b1_ratio = b1_total / float(b1_total + b2_total)

            # episode_length_buf increments before reward calculation in RL envs.
            t0_mask = env.episode_length_buf == 1
            t3_mask = env.episode_length_buf == 4
            t0_zero = t0_mean = float("nan")
            t3_zero = t3_mean = float("nan")
            if contact_counts is not None and t0_mask.any():
                t0_counts = contact_counts[t0_mask]
                t0_zero = (t0_counts == 0).float().mean().item() * 100.0
                t0_mean = t0_counts.mean().item()
            if contact_counts is not None and t3_mask.any():
                t3_counts = contact_counts[t3_mask]
                t3_zero = (t3_counts == 0).float().mean().item() * 100.0
                t3_mean = t3_counts.mean().item()

            try:
                obj = env.scene["object"]
            except KeyError:
                obj = None
            mass_env0 = float("nan")
            if obj is not None:
                try:
                    mass_env0 = obj.root_physx_view.get_masses()[0].mean().item()
                except Exception:
                    mass_env0 = float("nan")
            contact_force_env0 = float("nan")
            try:
                contact_force_env0 = force_magnitudes[0].sum().item()
            except Exception:
                contact_force_env0 = float("nan")
            lh_tgt_mean = float("nan")
            lh_err_mean = float("nan")
            lh_raw_mean = float("nan")
            try:
                action_term = env.action_manager.get_term("left_hand_action")
                robot = env.scene["robot"]
                joint_ids = action_term._joint_ids
                targets = action_term.processed_actions
                joint_pos = robot.data.joint_pos
                if isinstance(joint_ids, slice):
                    cur = joint_pos[:, joint_ids]
                else:
                    joint_ids_t = torch.as_tensor(joint_ids, device=joint_pos.device)
                    cur = joint_pos.index_select(1, joint_ids_t)
                err = (targets - cur).abs()
                lh_tgt_mean = targets[0].mean().item()
                lh_err_mean = err[0].mean().item()
                lh_raw_mean = action_term.raw_actions[0].mean().item()
            except Exception:
                lh_tgt_mean = float("nan")
                lh_err_mean = float("nan")
                lh_raw_mean = float("nan")

            if DEBUG_REWARD_LOGS:
                print(
                    "DBG_B_REWARD "
                f"step={debug_step} "
                f"contact_zero_pct={contact_zero_pct:.1f} "
                f"contact_mean={contact_mean:.3f} "
                f"slip_mean={slip_mean:.4f} "
                f"slip_c0={slip_c0:.4f} "
                f"slip_c1={slip_c1:.4f} "
                f"force_mean={force_mean:.4f} "
                f"overgrip_mean={overgrip_mean:.4f} "
                f"t0_zero_pct={t0_zero:.1f} "
                f"t0_mean={t0_mean:.3f} "
                f"t3_zero_pct={t3_zero:.1f} "
                f"t3_mean={t3_mean:.3f} "
                f"mass_env0={mass_env0:.3f} "
                f"contact_force_env0={contact_force_env0:.4f} "
                f"lh_tgt_mean={lh_tgt_mean:.4f} "
                f"lh_err_mean={lh_err_mean:.4f} "
                f"lh_raw_mean={lh_raw_mean:.4f} "
                f"b1_ratio={b1_ratio:.3f} "
                f"slip_active={slip_active} "
                f"force_active={force_active} "
                f"overgrip_active={overgrip_active}"
            )
            setattr(env, "_debug_reward_step", debug_step + 1)

    return penalty


def grip_force_in_band(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    force_scale: float = 20.0,
) -> torch.Tensor:
    """Reward for keeping grip force within commanded band.

    Uses command as (force_low, force_high) band.

    Args:
        env: Environment instance
        sensor_cfg: Contact sensor configuration
        command_name: Name of grip margin command
        force_scale: Scale factor to convert command to force (N)

    Returns:
        Band reward in [0, 1] (num_envs,)
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Get total grip force
    contact_forces = contact_sensor.data.net_forces_w
    force_magnitudes = torch.norm(contact_forces, dim=-1)

    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]

    total_force = force_magnitudes.sum(dim=-1)

    # Extract force band from command (using x and y as low/high)
    force_low = command[:, 0] * force_scale
    force_high = command[:, 1] * force_scale

    # Check if within band
    in_band = (total_force >= force_low) & (total_force <= force_high)

    # Compute distance from band for smooth reward
    below_band = torch.clamp(force_low - total_force, 0.0, None)
    above_band = torch.clamp(total_force - force_high, 0.0, None)
    distance = below_band + above_band

    # Smooth reward with tanh
    reward = 1.0 - torch.tanh(distance / force_scale)

    return reward


def grip_force_in_band_multi(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    command_name: str,
    force_scale: float = 20.0,
) -> torch.Tensor:
    """Reward for keeping grip force within commanded band (multi-sensor)."""
    command = env.command_manager.get_command(command_name)
    force_magnitudes = _contact_force_magnitudes(env, sensor_names)
    total_force = force_magnitudes.sum(dim=-1)
    force_low = command[:, 0] * force_scale
    force_high = command[:, 1] * force_scale
    in_band = (total_force >= force_low) & (total_force <= force_high)
    below_band = torch.clamp(force_low - total_force, 0.0, None)
    above_band = torch.clamp(total_force - force_high, 0.0, None)
    distance = below_band + above_band
    reward = 1.0 - torch.tanh(distance / force_scale)
    return reward
