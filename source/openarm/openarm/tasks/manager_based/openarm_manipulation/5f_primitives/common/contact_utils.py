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

"""Contact sensing utilities for dexterous manipulation primitives.

This module provides functions for:
- Contact flag detection per fingertip
- Normal force proxy computation
- Slip (tangential velocity) proxy computation
- Contact persistence metrics
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Tuple, Optional

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

__all__ = [
    "get_contact_flags",
    "get_normal_force_proxy",
    "get_slip_proxy",
    "get_slip_direction_components",
    "compute_contact_persistence",
    "detect_object_drop",
]


def get_contact_flags(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Get binary contact flags for each contact link.

    Args:
        env: The environment instance
        sensor_cfg: Scene entity config for the contact sensor
        threshold: Force threshold for considering contact active (N)

    Returns:
        Boolean tensor of shape (num_envs, num_contact_links)
    """
    # TODO: Adapt based on actual contact sensor implementation in IsaacLab
    # This assumes a ContactSensor is configured in the scene
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # Get net contact forces per body
    # Shape: (num_envs, num_bodies, 3)
    contact_forces = contact_sensor.data.net_forces_w

    # Compute force magnitudes
    force_magnitudes = torch.norm(contact_forces, dim=-1)

    # Select only the contact links specified in sensor_cfg
    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]

    # Threshold to get binary contact flags
    contact_flags = force_magnitudes > threshold

    return contact_flags


def get_normal_force_proxy(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    surface_normal: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Get normal force proxy for each contact link.

    If contact normal data is available, projects forces onto normal direction.
    Otherwise, uses total force magnitude as proxy.

    Args:
        env: The environment instance
        sensor_cfg: Scene entity config for the contact sensor
        surface_normal: Optional surface normal direction (num_envs, 3) or (3,)
                       If None, uses force magnitude as proxy

    Returns:
        Tensor of shape (num_envs, num_contact_links) with normal force values
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w

    if sensor_cfg.body_ids is not None:
        contact_forces = contact_forces[:, sensor_cfg.body_ids, :]

    if surface_normal is not None:
        # Project forces onto surface normal
        # Normalize the surface normal
        if surface_normal.dim() == 1:
            surface_normal = surface_normal.unsqueeze(0).expand(contact_forces.shape[0], -1)

        normal = surface_normal / (torch.norm(surface_normal, dim=-1, keepdim=True) + 1e-8)
        # Dot product: (num_envs, num_links, 3) . (num_envs, 1, 3) -> (num_envs, num_links)
        normal_forces = torch.einsum('ijk,ik->ij', contact_forces, normal)
        # Take absolute value (normal force is positive into surface)
        return torch.abs(normal_forces)
    else:
        # Use force magnitude as proxy
        return torch.norm(contact_forces, dim=-1)


def get_slip_proxy(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: Optional[SceneEntityCfg] = None,
) -> torch.Tensor:
    """Get slip (tangential relative velocity) proxy for each contact link.

    Computes relative velocity between fingertips and object (or world if no object).

    Args:
        env: The environment instance
        robot_cfg: Scene entity config for the robot (with body_ids for contact links)
        object_cfg: Optional scene entity config for the grasped object

    Returns:
        Tensor of shape (num_envs, num_contact_links, 3) with slip velocity vectors
    """
    robot = env.scene[robot_cfg.name]

    # Get fingertip velocities
    # Shape: (num_envs, num_bodies, 3)
    fingertip_vel = robot.data.body_lin_vel_w

    if robot_cfg.body_ids is not None:
        fingertip_vel = fingertip_vel[:, robot_cfg.body_ids, :]

    if object_cfg is not None:
        # Get object velocity and compute relative velocity
        obj = env.scene[object_cfg.name]
        obj_vel = obj.data.root_lin_vel_w  # (num_envs, 3)
        # Expand for broadcasting
        obj_vel = obj_vel.unsqueeze(1)  # (num_envs, 1, 3)
        slip_velocity = fingertip_vel - obj_vel
    else:
        # Slip relative to world (stationary)
        slip_velocity = fingertip_vel

    return slip_velocity


def get_slip_direction_components(
    slip_velocity: torch.Tensor,
    allowed_direction: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose slip velocity into allowed and non-allowed components.

    Args:
        slip_velocity: Slip velocity tensor (num_envs, num_links, 3) or (num_envs, 3)
        allowed_direction: Unit vector for allowed slip direction (num_envs, 3) or (3,)

    Returns:
        Tuple of:
        - parallel_component: Magnitude of slip in allowed direction (num_envs, [num_links])
        - perpendicular_component: Magnitude of slip perpendicular to allowed direction
    """
    # Normalize allowed direction
    if allowed_direction.dim() == 1:
        allowed_direction = allowed_direction.unsqueeze(0)

    allowed_dir_norm = allowed_direction / (torch.norm(allowed_direction, dim=-1, keepdim=True) + 1e-8)

    if slip_velocity.dim() == 3:
        # Per-link slip velocity
        # allowed_dir_norm: (num_envs, 3) -> (num_envs, 1, 3)
        allowed_dir_norm = allowed_dir_norm.unsqueeze(1)

        # Parallel component (projection onto allowed direction)
        parallel_magnitude = torch.einsum('ijk,ijk->ij', slip_velocity, allowed_dir_norm.expand_as(slip_velocity))

        # Parallel velocity vector
        parallel_velocity = parallel_magnitude.unsqueeze(-1) * allowed_dir_norm

        # Perpendicular velocity vector
        perpendicular_velocity = slip_velocity - parallel_velocity
        perpendicular_magnitude = torch.norm(perpendicular_velocity, dim=-1)

    else:
        # Single slip velocity per env
        parallel_magnitude = torch.sum(slip_velocity * allowed_dir_norm, dim=-1)
        parallel_velocity = parallel_magnitude.unsqueeze(-1) * allowed_dir_norm
        perpendicular_velocity = slip_velocity - parallel_velocity
        perpendicular_magnitude = torch.norm(perpendicular_velocity, dim=-1)

    return parallel_magnitude, perpendicular_magnitude


def compute_contact_persistence(
    contact_flags: torch.Tensor,
    prev_contact_flags: torch.Tensor,
    required_contacts: int = 3,
) -> torch.Tensor:
    """Compute contact persistence metric.

    Rewards maintaining stable contacts over time.

    Args:
        contact_flags: Current contact flags (num_envs, num_links)
        prev_contact_flags: Previous step contact flags (num_envs, num_links)
        required_contacts: Minimum number of contacts required for stable grasp

    Returns:
        Tensor of shape (num_envs,) with persistence scores [0, 1]
    """
    num_contacts = contact_flags.sum(dim=-1).float()
    prev_num_contacts = prev_contact_flags.sum(dim=-1).float()

    # Contacts that remained stable
    stable_contacts = (contact_flags & prev_contact_flags).sum(dim=-1).float()

    # Persistence ratio
    max_contacts = torch.maximum(num_contacts, prev_num_contacts)
    persistence = stable_contacts / (max_contacts + 1e-8)

    # Scale by whether we meet minimum contact requirement
    meets_minimum = (num_contacts >= required_contacts).float()

    return persistence * meets_minimum


def detect_object_drop(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    min_height: float = 0.0,
    contact_sensor_cfg: Optional[SceneEntityCfg] = None,
    min_contacts: int = 2,
) -> torch.Tensor:
    """Detect if object has been dropped.

    Uses combination of height threshold and contact loss.

    Args:
        env: The environment instance
        object_cfg: Scene entity config for the object
        min_height: Minimum height threshold (m)
        contact_sensor_cfg: Optional contact sensor config
        min_contacts: Minimum contacts required to consider object held

    Returns:
        Boolean tensor of shape (num_envs,) indicating drop events
    """
    obj = env.scene[object_cfg.name]
    object_height = obj.data.root_pos_w[:, 2]

    # Height-based drop detection
    height_dropped = object_height < min_height

    if contact_sensor_cfg is not None:
        # Contact-based drop detection
        contact_flags = get_contact_flags(env, contact_sensor_cfg)
        num_contacts = contact_flags.sum(dim=-1)
        contact_lost = num_contacts < min_contacts
        return height_dropped | contact_lost

    return height_dropped
