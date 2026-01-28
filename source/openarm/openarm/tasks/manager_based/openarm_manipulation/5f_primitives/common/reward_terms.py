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

"""Common reward terms for dexterous manipulation primitives.

This module provides reward functions that are shared across multiple primitives:
- Pose tracking rewards (for primitive a)
- Contact and force rewards (for primitives b, c)
- Synergy tracking rewards (for primitive d)
- Safety and smoothness penalties
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude, combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    # Pose tracking (primitive a)
    "ee_position_tracking_l2",
    "ee_position_tracking_tanh",
    "ee_orientation_tracking",
    "ee_velocity_penalty",
    # Contact rewards (primitives b, c)
    "contact_persistence_reward",
    "slip_penalty",
    "slip_band_reward",
    "normal_force_stability",
    "force_spike_penalty",
    "overgrip_penalty",
    # Synergy rewards (primitive d)
    "synergy_tracking_error",
    # Safety penalties (all primitives)
    "joint_limit_penalty",
    "self_collision_penalty",
    "action_smoothness_penalty",
]


# ==============================================================================
# Pose Tracking Rewards (Primitive A)
# ==============================================================================


def ee_position_tracking_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize position tracking error using L2 norm.

    Args:
        env: Environment instance
        command_name: Name of the pose command
        asset_cfg: Robot configuration with body_names for EE

    Returns:
        L2 position error (num_envs,)
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Desired position in body frame
    des_pos_b = command[:, :3]
    # Transform to world frame
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )

    # Current EE position
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :]

    return torch.norm(curr_pos_w - des_pos_w, dim=-1)


def ee_position_tracking_tanh(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.1,
) -> torch.Tensor:
    """Reward position tracking using tanh kernel for smooth gradient.

    Args:
        env: Environment instance
        command_name: Name of the pose command
        asset_cfg: Robot configuration with body_names for EE
        std: Standard deviation for tanh kernel

    Returns:
        Tanh-shaped reward in [0, 1] (num_envs,)
    """
    distance = ee_position_tracking_l2(env, command_name, asset_cfg)
    return 1.0 - torch.tanh(distance / std)


def ee_orientation_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize orientation tracking error using quaternion error.

    Args:
        env: Environment instance
        command_name: Name of the pose command
        asset_cfg: Robot configuration with body_names for EE

    Returns:
        Orientation error in radians (num_envs,)
    """
    from isaaclab.utils.math import quat_mul

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Desired orientation in body frame
    des_quat_b = command[:, 3:7]
    # Transform to world frame
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)

    # Current EE orientation
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]

    return quat_error_magnitude(curr_quat_w, des_quat_w)


def ee_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize residual end-effector velocity (for settling near target).

    Args:
        env: Environment instance
        asset_cfg: Robot configuration with body_ids for EE

    Returns:
        EE velocity magnitude (num_envs,)
    """
    asset = env.scene[asset_cfg.name]

    # Get EE linear and angular velocity
    lin_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :]
    ang_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0], :]

    # Combined velocity magnitude
    vel_magnitude = torch.norm(lin_vel, dim=-1) + 0.1 * torch.norm(ang_vel, dim=-1)

    return vel_magnitude


# ==============================================================================
# Contact Rewards (Primitives B, C)
# ==============================================================================


def contact_persistence_reward(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    required_contacts: int = 3,
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward maintaining stable contacts over time.

    Args:
        env: Environment instance
        contact_sensor_cfg: Contact sensor configuration
        required_contacts: Minimum contacts for stable grasp
        contact_threshold: Force threshold for contact detection

    Returns:
        Persistence reward in [0, 1] (num_envs,)
    """
    from .contact_utils import get_contact_flags

    contact_flags = get_contact_flags(env, contact_sensor_cfg, contact_threshold)
    num_contacts = contact_flags.sum(dim=-1).float()

    # Reward scales with number of contacts up to required
    contact_ratio = torch.clamp(num_contacts / required_contacts, 0.0, 1.0)

    return contact_ratio


def slip_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = None,
    max_slip: float = 0.1,
) -> torch.Tensor:
    """Penalize tangential slip at contact points.

    Args:
        env: Environment instance
        robot_cfg: Robot config with body_ids for contact links
        object_cfg: Optional object config for relative velocity
        max_slip: Maximum acceptable slip velocity (m/s)

    Returns:
        Normalized slip penalty (num_envs,)
    """
    from .contact_utils import get_slip_proxy

    slip_velocity = get_slip_proxy(env, robot_cfg, object_cfg)

    # Compute slip magnitude per contact link
    slip_magnitude = torch.norm(slip_velocity, dim=-1)  # (num_envs, num_links)

    # Average slip across links
    avg_slip = slip_magnitude.mean(dim=-1)

    # Normalize by max acceptable slip
    return torch.clamp(avg_slip / max_slip, 0.0, 1.0)


def slip_band_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    allowed_direction: torch.Tensor,
    min_slip: float = 0.0,
    max_slip: float = 0.05,
    object_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Reward controlled slip within a band in allowed direction.

    For primitive C: allows smooth slip in one direction while penalizing
    slip in other directions.

    Args:
        env: Environment instance
        robot_cfg: Robot config with body_ids for contact links
        allowed_direction: Unit vector for allowed slip direction (num_envs, 3)
        min_slip: Minimum desired slip speed
        max_slip: Maximum desired slip speed
        object_cfg: Optional object config

    Returns:
        Band reward in [0, 1] (num_envs,)
    """
    from .contact_utils import get_slip_proxy, get_slip_direction_components

    slip_velocity = get_slip_proxy(env, robot_cfg, object_cfg)

    # Average slip across contact links
    avg_slip = slip_velocity.mean(dim=1)  # (num_envs, 3)

    # Decompose into allowed and non-allowed components
    parallel_mag, perp_mag = get_slip_direction_components(avg_slip, allowed_direction)

    # Reward if parallel slip is within band
    in_band = (parallel_mag >= min_slip) & (parallel_mag <= max_slip)

    # Penalize perpendicular slip
    perp_penalty = torch.clamp(perp_mag / max_slip, 0.0, 1.0)

    # Combined reward
    band_reward = in_band.float() * (1.0 - perp_penalty)

    return band_reward


def normal_force_stability(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward stable normal forces (low variance over time).

    Requires storing previous force values in env.

    Args:
        env: Environment instance
        contact_sensor_cfg: Contact sensor configuration

    Returns:
        Force stability reward (num_envs,)
    """
    from .contact_utils import get_normal_force_proxy

    normal_forces = get_normal_force_proxy(env, contact_sensor_cfg)

    # Get previous forces if available
    if hasattr(env, '_prev_normal_forces'):
        prev_forces = env._prev_normal_forces
        force_change = torch.abs(normal_forces - prev_forces)
        stability = 1.0 / (1.0 + force_change.mean(dim=-1))
    else:
        stability = torch.ones(env.num_envs, device=env.device)

    # Store for next step
    env._prev_normal_forces = normal_forces.clone()

    return stability


def force_spike_penalty(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    spike_threshold: float = 5.0,
) -> torch.Tensor:
    """Penalize sudden force spikes (impacts).

    Args:
        env: Environment instance
        contact_sensor_cfg: Contact sensor configuration
        spike_threshold: Force rate threshold for spike detection (N/s)

    Returns:
        Spike penalty (num_envs,)
    """
    from .contact_utils import get_normal_force_proxy

    normal_forces = get_normal_force_proxy(env, contact_sensor_cfg)

    if hasattr(env, '_prev_normal_forces_spike'):
        prev_forces = env._prev_normal_forces_spike
        # Force rate of change
        force_rate = torch.abs(normal_forces - prev_forces) / env.step_dt
        # Max rate across all contacts
        max_rate = force_rate.max(dim=-1)[0]
        # Penalty for rates above threshold
        penalty = torch.clamp((max_rate - spike_threshold) / spike_threshold, 0.0, 1.0)
    else:
        penalty = torch.zeros(env.num_envs, device=env.device)

    env._prev_normal_forces_spike = normal_forces.clone()

    return penalty


def overgrip_penalty(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    target_force_range: tuple = (1.0, 10.0),
) -> torch.Tensor:
    """Penalize excessive grip force beyond target band.

    Args:
        env: Environment instance
        contact_sensor_cfg: Contact sensor configuration
        target_force_range: (min, max) target force range (N)

    Returns:
        Overgrip penalty (num_envs,)
    """
    from .contact_utils import get_normal_force_proxy

    normal_forces = get_normal_force_proxy(env, contact_sensor_cfg)
    total_force = normal_forces.sum(dim=-1)

    min_force, max_force = target_force_range

    # Undergrip penalty
    undergrip = torch.clamp(min_force - total_force, 0.0, min_force) / min_force

    # Overgrip penalty
    overgrip = torch.clamp(total_force - max_force, 0.0, max_force) / max_force

    return undergrip + overgrip


# ==============================================================================
# Synergy Rewards (Primitive D)
# ==============================================================================


def synergy_tracking_error(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    synergy_matrix: torch.Tensor,
    command_name: str,
) -> torch.Tensor:
    """Compute error between current hand pose and target synergy.

    Args:
        env: Environment instance
        robot_cfg: Robot config with joint_names for hand
        synergy_matrix: Synergy basis (num_synergies, num_joints)
        command_name: Name of synergy coefficient command

    Returns:
        Synergy error (num_envs,)
    """
    from .frames_utils import compute_synergy_error

    robot = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Get hand joint positions
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]

    # Target synergy coefficients from command
    target_coeffs = command  # Assumes command is synergy coefficients

    return compute_synergy_error(joint_pos, target_coeffs, synergy_matrix)


# ==============================================================================
# Safety Penalties (All Primitives)
# ==============================================================================


def joint_limit_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    soft_limit_factor: float = 0.9,
) -> torch.Tensor:
    """Penalize joints approaching their limits.

    Args:
        env: Environment instance
        asset_cfg: Asset configuration
        soft_limit_factor: Factor of limit at which penalty starts

    Returns:
        Joint limit penalty (num_envs,)
    """
    asset = env.scene[asset_cfg.name]

    # Get joint positions and limits
    joint_pos = asset.data.joint_pos
    if asset_cfg.joint_ids is not None:
        joint_pos = joint_pos[:, asset_cfg.joint_ids]

    # Get joint limits from asset
    joint_limits = asset.data.soft_joint_pos_limits
    if asset_cfg.joint_ids is not None:
        joint_limits = joint_limits[:, asset_cfg.joint_ids, :]

    lower_limit = joint_limits[..., 0]
    upper_limit = joint_limits[..., 1]

    # Compute margin from limits
    range_size = upper_limit - lower_limit
    soft_lower = lower_limit + (1 - soft_limit_factor) * range_size
    soft_upper = upper_limit - (1 - soft_limit_factor) * range_size

    # Penalty for being outside soft limits
    below_penalty = torch.clamp(soft_lower - joint_pos, 0.0, None)
    above_penalty = torch.clamp(joint_pos - soft_upper, 0.0, None)

    # Sum penalties across joints
    total_penalty = (below_penalty + above_penalty).sum(dim=-1)

    return total_penalty


def self_collision_penalty(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize self-collisions between robot links.

    Args:
        env: Environment instance
        contact_sensor_cfg: Contact sensor configuration for self-collision

    Returns:
        Self-collision penalty (num_envs,)
    """
    # TODO: Implement based on IsaacLab self-collision detection
    # This requires checking contact pairs between robot's own bodies
    contact_sensor = env.scene[contact_sensor_cfg.name]

    # Check if there are any self-contacts
    # The contact sensor should be configured to detect self-collisions
    contact_forces = contact_sensor.data.net_forces_w
    force_magnitudes = torch.norm(contact_forces, dim=-1)

    # Sum of all self-contact forces
    total_self_contact = force_magnitudes.sum(dim=-1)

    return total_self_contact


def action_smoothness_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = None,
) -> torch.Tensor:
    """Penalize rapid changes in actions (action rate).

    Args:
        env: Environment instance
        action_name: Optional specific action to check

    Returns:
        Action rate penalty (num_envs,)
    """
    # Get current and previous actions
    if action_name is not None:
        current_action = env.action_manager.get_term(action_name).processed_actions
        if hasattr(env, f'_prev_action_{action_name}'):
            prev_action = getattr(env, f'_prev_action_{action_name}')
        else:
            prev_action = current_action.clone()
        setattr(env, f'_prev_action_{action_name}', current_action.clone())
    else:
        current_action = env.action_manager.action
        if hasattr(env, '_prev_action'):
            prev_action = env._prev_action
        else:
            prev_action = current_action.clone()
        env._prev_action = current_action.clone()

    # Compute action rate
    action_diff = current_action - prev_action
    action_rate = torch.norm(action_diff, dim=-1)

    return action_rate
