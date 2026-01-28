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

"""Termination functions for Primitive B: Contact Force Hold."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


__all__ = [
    "all_contacts_lost",
    "all_contacts_lost_multi",
    "sustained_large_slip",
    "stable_grasp_achieved",
]


def all_contacts_lost(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    num_steps: int = 20,
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """Terminate if all contacts lost for sustained period.

    Args:
        env: Environment instance
        left_sensor_cfg: Left hand contact sensor
        right_sensor_cfg: Right hand contact sensor
        num_steps: Number of steps to sustain before termination
        contact_threshold: Force threshold for contact

    Returns:
        Boolean termination flags (num_envs,)
    """
    left_sensor: ContactSensor = env.scene[left_sensor_cfg.name]
    right_sensor: ContactSensor = env.scene[right_sensor_cfg.name]

    # Get contact forces
    left_forces = torch.norm(left_sensor.data.net_forces_w, dim=-1)
    right_forces = torch.norm(right_sensor.data.net_forces_w, dim=-1)

    # Count contacts
    left_contacts = (left_forces > contact_threshold).sum(dim=-1)
    right_contacts = (right_forces > contact_threshold).sum(dim=-1)
    total_contacts = left_contacts + right_contacts

    # Track consecutive steps with no contacts
    if not hasattr(env, '_no_contact_counter'):
        env._no_contact_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    no_contacts = total_contacts == 0
    env._no_contact_counter = torch.where(
        no_contacts,
        env._no_contact_counter + 1,
        torch.zeros_like(env._no_contact_counter)
    )

    return env._no_contact_counter >= num_steps


def all_contacts_lost_multi(
    env: ManagerBasedRLEnv,
    left_sensor_names: list[str],
    right_sensor_names: list[str],
    num_steps: int = 20,
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """Terminate if all contacts lost for sustained period (multi-sensor)."""
    left_forces = []
    right_forces = []
    for sensor_name in left_sensor_names:
        sensor: ContactSensor = env.scene[sensor_name]
        mags = torch.norm(sensor.data.net_forces_w, dim=-1)
        if mags.dim() == 2:
            mags = mags.max(dim=-1)[0]
        left_forces.append(mags)
    for sensor_name in right_sensor_names:
        sensor: ContactSensor = env.scene[sensor_name]
        mags = torch.norm(sensor.data.net_forces_w, dim=-1)
        if mags.dim() == 2:
            mags = mags.max(dim=-1)[0]
        right_forces.append(mags)

    if left_forces:
        left_forces = torch.stack(left_forces, dim=-1)
        left_contacts = (left_forces > contact_threshold).sum(dim=-1)
    else:
        left_contacts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    if right_forces:
        right_forces = torch.stack(right_forces, dim=-1)
        right_contacts = (right_forces > contact_threshold).sum(dim=-1)
    else:
        right_contacts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    total_contacts = left_contacts + right_contacts

    if not hasattr(env, "_no_contact_counter"):
        env._no_contact_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    no_contacts = total_contacts == 0
    env._no_contact_counter = torch.where(
        no_contacts,
        env._no_contact_counter + 1,
        torch.zeros_like(env._no_contact_counter),
    )

    return env._no_contact_counter >= num_steps


def sustained_large_slip(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    slip_threshold: float = 0.1,
    num_steps: int = 10,
) -> torch.Tensor:
    """Terminate if large slip sustained for multiple steps.

    Args:
        env: Environment instance
        robot_cfg: Robot config with contact link bodies
        object_cfg: Object config
        slip_threshold: Slip velocity threshold (m/s)
        num_steps: Steps to sustain before termination

    Returns:
        Boolean termination flags (num_envs,)
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    # Compute slip velocity
    link_vel = robot.data.body_lin_vel_w
    if robot_cfg.body_ids is not None:
        link_vel = link_vel[:, robot_cfg.body_ids, :]

    obj_vel = obj.data.root_lin_vel_w.unsqueeze(1)
    relative_vel = link_vel - obj_vel
    slip_mag = torch.norm(relative_vel, dim=-1).max(dim=-1)[0]

    # Track consecutive high-slip steps
    counter_name = f"_large_slip_counter_{robot_cfg.name}"
    if not hasattr(env, counter_name):
        setattr(env, counter_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.long))

    counter = getattr(env, counter_name)
    large_slip = slip_mag > slip_threshold
    counter = torch.where(large_slip, counter + 1, torch.zeros_like(counter))
    setattr(env, counter_name, counter)

    return counter >= num_steps


def stable_grasp_achieved(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    min_contacts: int = 4,
    max_slip: float = 0.02,
    num_steps: int = 30,
) -> torch.Tensor:
    """Success termination when stable grasp is achieved.

    Args:
        env: Environment instance
        left_sensor_cfg: Left contact sensor
        right_sensor_cfg: Right contact sensor
        robot_cfg: Robot config
        object_cfg: Object config
        min_contacts: Minimum total contacts
        max_slip: Maximum allowed slip velocity
        num_steps: Steps to maintain stability

    Returns:
        Boolean success flags (num_envs,)
    """
    left_sensor: ContactSensor = env.scene[left_sensor_cfg.name]
    right_sensor: ContactSensor = env.scene[right_sensor_cfg.name]
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    # Count contacts
    left_forces = torch.norm(left_sensor.data.net_forces_w, dim=-1)
    right_forces = torch.norm(right_sensor.data.net_forces_w, dim=-1)
    total_contacts = (left_forces > 0.1).sum(dim=-1) + (right_forces > 0.1).sum(dim=-1)

    # Compute slip
    link_vel = robot.data.body_lin_vel_w
    if robot_cfg.body_ids is not None:
        link_vel = link_vel[:, robot_cfg.body_ids, :]
    obj_vel = obj.data.root_lin_vel_w.unsqueeze(1)
    slip_mag = torch.norm(link_vel - obj_vel, dim=-1).max(dim=-1)[0]

    # Check stability criteria
    is_stable = (total_contacts >= min_contacts) & (slip_mag < max_slip)

    # Track consecutive stable steps
    if not hasattr(env, '_stable_grasp_counter'):
        env._stable_grasp_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    env._stable_grasp_counter = torch.where(
        is_stable,
        env._stable_grasp_counter + 1,
        torch.zeros_like(env._stable_grasp_counter)
    )

    return env._stable_grasp_counter >= num_steps
