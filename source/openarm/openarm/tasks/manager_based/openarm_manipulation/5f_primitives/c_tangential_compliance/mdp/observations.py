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

"""Observation functions for Primitive C: Tangential Compliance."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Import from primitive b
from ...b_contact_force_hold.mdp.observations import contact_flags, normal_force_magnitude

__all__ = [
    "contact_flags",
    "normal_force_magnitude",
    "slip_velocity_vector",
]


def slip_velocity_vector(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Get slip velocity vectors (not just magnitude).

    Args:
        env: Environment instance
        robot_cfg: Robot config with body_names for contact links
        object_cfg: Optional object config for relative velocity

    Returns:
        Slip velocity vectors flattened (num_envs, num_bodies * 3)
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

    # Flatten: (num_envs, num_links, 3) -> (num_envs, num_links * 3)
    return relative_vel.reshape(env.num_envs, -1)
