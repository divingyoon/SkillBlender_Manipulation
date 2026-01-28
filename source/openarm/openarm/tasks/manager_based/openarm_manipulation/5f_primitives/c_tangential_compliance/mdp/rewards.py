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

"""Reward functions for Primitive C: Tangential Compliance."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

# Import from primitive b
from ...b_contact_force_hold.mdp.rewards import contact_persistence_reward

__all__ = [
    "contact_persistence_reward",
    "nonallowed_slip_penalty",
    "slip_in_allowed_band",
    "force_variance_penalty",
]


def nonallowed_slip_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Penalty for slip perpendicular to allowed direction.

    Args:
        env: Environment instance
        robot_cfg: Robot config with contact link body_names
        object_cfg: Object config
        command_name: Name of slip direction command

    Returns:
        Perpendicular slip penalty (num_envs,)
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Get allowed direction from command (first 3 elements)
    allowed_dir = command[:, :3]
    allowed_dir = allowed_dir / (torch.norm(allowed_dir, dim=-1, keepdim=True) + 1e-8)

    # Get slip velocity
    link_vel = robot.data.body_lin_vel_w
    if robot_cfg.body_ids is not None:
        link_vel = link_vel[:, robot_cfg.body_ids, :]
    obj_vel = obj.data.root_lin_vel_w.unsqueeze(1)
    slip_vel = link_vel - obj_vel

    # Average slip across links
    avg_slip = slip_vel.mean(dim=1)  # (num_envs, 3)

    # Project onto allowed direction
    parallel_mag = torch.sum(avg_slip * allowed_dir, dim=-1, keepdim=True)
    parallel_vel = parallel_mag * allowed_dir

    # Perpendicular component
    perp_vel = avg_slip - parallel_vel
    perp_mag = torch.norm(perp_vel, dim=-1)

    # Normalize penalty
    return torch.clamp(perp_mag / 0.1, 0.0, 1.0)


def slip_in_allowed_band(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    command_name: str,
    min_slip: float = 0.0,
    max_slip: float = 0.05,
) -> torch.Tensor:
    """Reward for slip in allowed direction within speed band.

    Args:
        env: Environment instance
        robot_cfg: Robot config
        object_cfg: Object config
        command_name: Slip direction command name
        min_slip: Minimum desired slip speed
        max_slip: Maximum desired slip speed

    Returns:
        Band reward in [0, 1] (num_envs,)
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Get allowed direction
    allowed_dir = command[:, :3]
    allowed_dir = allowed_dir / (torch.norm(allowed_dir, dim=-1, keepdim=True) + 1e-8)

    # Get slip velocity
    link_vel = robot.data.body_lin_vel_w
    if robot_cfg.body_ids is not None:
        link_vel = link_vel[:, robot_cfg.body_ids, :]
    obj_vel = obj.data.root_lin_vel_w.unsqueeze(1)
    slip_vel = link_vel - obj_vel

    # Average slip
    avg_slip = slip_vel.mean(dim=1)

    # Project onto allowed direction
    parallel_mag = torch.sum(avg_slip * allowed_dir, dim=-1)

    # Check if within band
    in_band = (parallel_mag >= min_slip) & (parallel_mag <= max_slip)

    # Smooth reward: 1 if in band, tanh falloff otherwise
    below = torch.clamp(min_slip - parallel_mag, 0.0, None)
    above = torch.clamp(parallel_mag - max_slip, 0.0, None)
    distance = below + above

    reward = 1.0 - torch.tanh(distance / max_slip)

    return reward


def force_variance_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalty for force variance (instability during slip).

    Args:
        env: Environment instance
        sensor_cfg: Contact sensor config

    Returns:
        Force variance penalty (num_envs,)
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # Get force magnitudes
    contact_forces = contact_sensor.data.net_forces_w
    force_magnitudes = torch.norm(contact_forces, dim=-1)

    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]

    # Store history for variance computation
    buffer_name = f"_force_history_{sensor_cfg.name}"
    history_len = 5

    if not hasattr(env, buffer_name):
        setattr(env, buffer_name, torch.zeros(
            env.num_envs, history_len, force_magnitudes.shape[-1],
            device=env.device
        ))

    history = getattr(env, buffer_name)
    # Shift and add new
    history = torch.roll(history, 1, dims=1)
    history[:, 0, :] = force_magnitudes
    setattr(env, buffer_name, history)

    # Compute variance across time
    variance = history.var(dim=1).mean(dim=-1)

    # Normalize
    return torch.clamp(variance / 5.0, 0.0, 1.0)
