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

"""Domain randomization utilities for dexterous manipulation primitives.

This module provides randomization functions for:
- Actuator parameters (latency, friction, stiffness)
- Object properties (mass, friction, CoM offset)
- Observation noise
- Contact parameters
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Dict, Tuple

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "randomize_actuator_stiffness_damping",
    "randomize_joint_friction",
    "add_actuator_latency",
    "randomize_object_mass",
    "randomize_object_friction",
    "randomize_object_com",
    "add_observation_noise",
]


def randomize_actuator_stiffness_damping(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    stiffness_range: Tuple[float, float] = (0.8, 1.2),
    damping_range: Tuple[float, float] = (0.8, 1.2),
) -> None:
    """Randomize actuator stiffness and damping within a range.

    Multiplies the default values by a random factor.

    Args:
        env: Environment instance
        env_ids: Environment indices to randomize
        asset_cfg: Asset configuration
        stiffness_range: (min, max) multiplier for stiffness
        damping_range: (min, max) multiplier for damping
    """
    asset = env.scene[asset_cfg.name]

    # Get default actuator properties
    # Note: This assumes actuator properties are stored in the asset
    # TODO: Verify the correct API for accessing actuator properties in IsaacLab

    num_envs = len(env_ids)
    device = env.device

    # Generate random multipliers
    stiffness_mult = torch.empty(num_envs, device=device).uniform_(*stiffness_range)
    damping_mult = torch.empty(num_envs, device=device).uniform_(*damping_range)

    # Apply multipliers
    # TODO: Implement actual actuator property modification
    # This depends on IsaacLab's API for runtime actuator modification
    pass


def randomize_joint_friction(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    friction_range: Tuple[float, float] = (0.5, 2.0),
) -> None:
    """Randomize joint friction coefficients.

    Args:
        env: Environment instance
        env_ids: Environment indices to randomize
        asset_cfg: Asset configuration
        friction_range: (min, max) friction coefficient range
    """
    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    device = env.device

    # Generate random friction values
    if asset_cfg.joint_ids is not None:
        num_joints = len(asset_cfg.joint_ids)
    else:
        num_joints = asset.num_joints

    friction_values = torch.empty(num_envs, num_joints, device=device).uniform_(*friction_range)

    # Apply friction values
    # TODO: Implement actual joint friction modification
    # This depends on IsaacLab/PhysX API for runtime joint property modification
    pass


def add_actuator_latency(
    env: ManagerBasedRLEnv,
    actions: torch.Tensor,
    latency_range: Tuple[float, float] = (0.0, 0.02),
    buffer_size: int = 5,
) -> torch.Tensor:
    """Add simulated actuator latency by buffering actions.

    Args:
        env: Environment instance
        actions: Current actions (num_envs, action_dim)
        latency_range: (min, max) latency in seconds
        buffer_size: Maximum buffer size for delayed actions

    Returns:
        Delayed actions (num_envs, action_dim)
    """
    device = env.device
    num_envs = actions.shape[0]

    # Initialize action buffer if not exists
    if not hasattr(env, '_action_buffer'):
        env._action_buffer = torch.zeros(
            num_envs, buffer_size, actions.shape[1], device=device
        )
        env._latency_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Randomize latency per environment (in simulation steps)
        dt = env.step_dt
        min_steps = int(latency_range[0] / dt)
        max_steps = min(int(latency_range[1] / dt), buffer_size - 1)
        env._latency_steps = torch.randint(min_steps, max_steps + 1, (num_envs,), device=device)

    # Shift buffer and add new action
    env._action_buffer = torch.roll(env._action_buffer, 1, dims=1)
    env._action_buffer[:, 0, :] = actions

    # Get delayed action based on latency
    delayed_actions = torch.zeros_like(actions)
    for i in range(num_envs):
        delay_idx = env._latency_steps[i].item()
        delayed_actions[i] = env._action_buffer[i, delay_idx]

    return delayed_actions


def randomize_object_mass(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    mass_range: Tuple[float, float] = (0.1, 2.0),
) -> None:
    """Randomize object mass.

    Args:
        env: Environment instance
        env_ids: Environment indices to randomize
        object_cfg: Object configuration
        mass_range: (min, max) mass in kg
    """
    obj = env.scene[object_cfg.name]
    num_envs = len(env_ids)
    device = env.device

    # Generate random masses
    masses = torch.empty(num_envs, device=device).uniform_(*mass_range)

    # Apply masses
    # TODO: Implement using IsaacLab's mass modification API
    # obj.write_root_physx_body_com_positions(...)
    pass


def randomize_object_friction(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    static_friction_range: Tuple[float, float] = (0.3, 1.5),
    dynamic_friction_range: Tuple[float, float] = (0.3, 1.5),
) -> None:
    """Randomize object friction coefficients.

    Args:
        env: Environment instance
        env_ids: Environment indices to randomize
        object_cfg: Object configuration
        static_friction_range: (min, max) static friction
        dynamic_friction_range: (min, max) dynamic friction
    """
    obj = env.scene[object_cfg.name]
    num_envs = len(env_ids)
    device = env.device

    # Generate random friction values
    static_friction = torch.empty(num_envs, device=device).uniform_(*static_friction_range)
    dynamic_friction = torch.empty(num_envs, device=device).uniform_(*dynamic_friction_range)

    # Ensure dynamic <= static
    dynamic_friction = torch.minimum(dynamic_friction, static_friction)

    # Apply friction
    # TODO: Implement using IsaacLab's material property API
    pass


def randomize_object_com(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    com_offset_range: Tuple[float, float] = (-0.02, 0.02),
) -> None:
    """Randomize object center of mass offset.

    Args:
        env: Environment instance
        env_ids: Environment indices to randomize
        object_cfg: Object configuration
        com_offset_range: (min, max) CoM offset in each axis (m)
    """
    obj = env.scene[object_cfg.name]
    num_envs = len(env_ids)
    device = env.device

    # Generate random CoM offsets (3D)
    com_offsets = torch.empty(num_envs, 3, device=device).uniform_(*com_offset_range)

    # Apply CoM offsets
    # TODO: Implement using IsaacLab's CoM modification API
    pass


def add_observation_noise(
    env: ManagerBasedRLEnv,
    observations: torch.Tensor,
    noise_std: float = 0.01,
    noise_type: str = "gaussian",
) -> torch.Tensor:
    """Add noise to observations for sim-to-real robustness.

    Args:
        env: Environment instance
        observations: Clean observations (num_envs, obs_dim)
        noise_std: Standard deviation of noise
        noise_type: Type of noise ("gaussian", "uniform")

    Returns:
        Noisy observations (num_envs, obs_dim)
    """
    device = env.device

    if noise_type == "gaussian":
        noise = torch.randn_like(observations) * noise_std
    elif noise_type == "uniform":
        noise = (torch.rand_like(observations) * 2 - 1) * noise_std
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return observations + noise


def reset_randomization_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Reset randomization-related state for specified environments.

    Call this during environment reset.

    Args:
        env: Environment instance
        env_ids: Environment indices to reset
    """
    # Reset action buffer if exists
    if hasattr(env, '_action_buffer'):
        env._action_buffer[env_ids] = 0.0

    # Reset latency (re-randomize)
    if hasattr(env, '_latency_steps'):
        device = env.device
        num_reset = len(env_ids)
        # Re-sample latencies for reset environments
        # Using stored range or default
        min_steps = 0
        max_steps = 2
        env._latency_steps[env_ids] = torch.randint(
            min_steps, max_steps + 1, (num_reset,), device=device
        )
