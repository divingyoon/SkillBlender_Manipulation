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

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.utils.stage import get_current_stage

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


__all__ = [
    "contact_flags",
    "contact_flags_multi",
    "normal_force_magnitude",
    "normal_force_magnitude_multi",
    "object_cross_section_size",
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


def object_cross_section_size(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    cylinder_radius: float,
    box_size_xy: tuple[float, float],
) -> torch.Tensor:
    """Return cylinder diameter or box cross-section diagonal based on spawned mesh type.

    For cylinders, returns 2 * radius * scale_xy.
    For boxes (Cube), returns sqrt((size_x*scale_xy)^2 + (size_y*scale_xy)^2).
    """
    cache = getattr(env, "_object_cross_section_cache", {})
    cache_key = (object_cfg.name, float(cylinder_radius), float(box_size_xy[0]), float(box_size_xy[1]))
    cached = cache.get(cache_key)
    if cached is not None and cached.shape[0] == env.num_envs:
        return cached

    stage = get_current_stage()
    obj = env.scene[object_cfg.name]
    prim_paths = sim_utils.find_matching_prim_paths(obj.cfg.prim_path)

    sizes = torch.zeros(env.num_envs, device=env.device)
    for env_id in range(env.num_envs):
        prim_path = prim_paths[env_id]
        mesh_prim = stage.GetPrimAtPath(prim_path + "/geometry/mesh")
        mesh_type = mesh_prim.GetTypeName() if mesh_prim.IsValid() else ""

        scale_xy = 1.0
        prim = stage.GetPrimAtPath(prim_path)
        scale_attr = prim.GetAttribute("xformOp:scale")
        if scale_attr is not None and scale_attr.HasAuthoredValue():
            scale_val = scale_attr.Get()
            try:
                scale_xy = float(scale_val[0])
            except Exception:
                scale_xy = 1.0

        if mesh_type == "Cylinder":
            size_val = 2.0 * cylinder_radius * scale_xy
        elif mesh_type == "Cube":
            size_x = box_size_xy[0] * scale_xy
            size_y = box_size_xy[1] * scale_xy
            size_val = math.sqrt(size_x * size_x + size_y * size_y)
        else:
            size_val = 0.0
        sizes[env_id] = size_val

    sizes = sizes.unsqueeze(-1)
    cache[cache_key] = sizes
    setattr(env, "_object_cross_section_cache", cache)
    return sizes
