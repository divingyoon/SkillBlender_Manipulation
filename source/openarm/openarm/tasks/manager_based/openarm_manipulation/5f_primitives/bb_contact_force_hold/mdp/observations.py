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

"""Observation functions for Primitive B: Contact Force Hold."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


__all__ = [
    "contact_flags",
    "contact_flags_multi",
    "normal_force_magnitude",
    "normal_force_magnitude_multi",
    "slip_velocity",
]


def _contact_force_magnitudes(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
) -> torch.Tensor:
    """Collect contact force magnitudes for multiple sensors.

    Returns:
        Force magnitudes (num_envs, num_sensors)
    """
    mags = []
    for sensor_name in sensor_names:
        contact_sensor: ContactSensor = env.scene[sensor_name]
        contact_forces = contact_sensor.data.net_forces_w
        force_magnitudes = torch.norm(contact_forces, dim=-1)
        if force_magnitudes.dim() == 2:
            force_magnitudes = force_magnitudes.max(dim=-1)[0]
        mags.append(force_magnitudes)
    return torch.stack(mags, dim=-1)


def contact_flags(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Get binary contact flags per contact body.

    Args:
        env: Environment instance
        sensor_cfg: Contact sensor configuration
        threshold: Force threshold for contact detection (N)

    Returns:
        Binary contact flags (num_envs, num_bodies)
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # Get net contact forces
    # Shape: (num_envs, num_bodies, 3)
    contact_forces = contact_sensor.data.net_forces_w

    # Compute force magnitudes
    force_magnitudes = torch.norm(contact_forces, dim=-1)

    # Select specific bodies if specified
    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]

    # Binary threshold
    flags = (force_magnitudes > threshold).float()

    return flags


def contact_flags_multi(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    threshold: float = 0.1,
) -> torch.Tensor:
    """Get binary contact flags for multiple single-body sensors.

    Args:
        env: Environment instance
        sensor_names: List of contact sensor names
        threshold: Force threshold for contact detection (N)

    Returns:
        Binary contact flags (num_envs, num_sensors)
    """
    force_magnitudes = _contact_force_magnitudes(env, sensor_names)
    return (force_magnitudes > threshold).float()


def normal_force_magnitude(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Get normal force magnitude per contact body.

    Uses total force magnitude as proxy for normal force.

    Args:
        env: Environment instance
        sensor_cfg: Contact sensor configuration

    Returns:
        Force magnitudes (num_envs, num_bodies)
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # Get net contact forces
    contact_forces = contact_sensor.data.net_forces_w

    # Compute magnitudes
    force_magnitudes = torch.norm(contact_forces, dim=-1)

    # Select specific bodies if specified
    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]

    return force_magnitudes


def normal_force_magnitude_multi(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
) -> torch.Tensor:
    """Get force magnitudes for multiple single-body sensors.

    Args:
        env: Environment instance
        sensor_names: List of contact sensor names

    Returns:
        Force magnitudes (num_envs, num_sensors)
    """
    return _contact_force_magnitudes(env, sensor_names)


def slip_velocity(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Get slip velocity (relative tangential velocity) per contact link.

    Args:
        env: Environment instance
        robot_cfg: Robot config with body_names for contact links
        object_cfg: Optional object config for relative velocity

    Returns:
        Slip velocity magnitudes (num_envs, num_bodies)
    """
    robot = env.scene[robot_cfg.name]

    # Get contact link velocities
    link_vel = robot.data.body_lin_vel_w

    if robot_cfg.body_ids is not None:
        link_vel = link_vel[:, robot_cfg.body_ids, :]

    if object_cfg is not None:
        # Compute relative velocity to object
        obj = env.scene[object_cfg.name]
        obj_vel = obj.data.root_lin_vel_w  # (num_envs, 3)
        obj_vel = obj_vel.unsqueeze(1)  # (num_envs, 1, 3)
        relative_vel = link_vel - obj_vel
    else:
        relative_vel = link_vel

    # Return velocity magnitude (slip proxy)
    slip_mag = torch.norm(relative_vel, dim=-1)

    return slip_mag
