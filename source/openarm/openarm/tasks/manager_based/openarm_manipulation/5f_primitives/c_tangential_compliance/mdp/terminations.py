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

"""Termination functions for Primitive C: Tangential Compliance."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Import from primitive b
from ...b_contact_force_hold.mdp.terminations import all_contacts_lost

__all__ = [
    "all_contacts_lost",
    "uncontrolled_slip_detected",
]


def uncontrolled_slip_detected(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    command_name: str,
    perp_threshold: float = 0.1,
    num_steps: int = 10,
) -> torch.Tensor:
    """Terminate if perpendicular slip exceeds threshold for sustained period.

    Args:
        env: Environment instance
        robot_cfg: Robot config
        object_cfg: Object config
        command_name: Slip direction command
        perp_threshold: Perpendicular slip threshold (m/s)
        num_steps: Steps to sustain before termination

    Returns:
        Boolean termination flags (num_envs,)
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
    slip_vel = (link_vel - obj_vel).mean(dim=1)

    # Compute perpendicular component
    parallel_mag = torch.sum(slip_vel * allowed_dir, dim=-1, keepdim=True)
    parallel_vel = parallel_mag * allowed_dir
    perp_vel = slip_vel - parallel_vel
    perp_mag = torch.norm(perp_vel, dim=-1)

    # Track consecutive violations
    if not hasattr(env, '_uncontrolled_slip_counter'):
        env._uncontrolled_slip_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    exceeds = perp_mag > perp_threshold
    env._uncontrolled_slip_counter = torch.where(
        exceeds,
        env._uncontrolled_slip_counter + 1,
        torch.zeros_like(env._uncontrolled_slip_counter)
    )

    return env._uncontrolled_slip_counter >= num_steps
