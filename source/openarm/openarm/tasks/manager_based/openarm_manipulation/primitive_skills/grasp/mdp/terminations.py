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

"""Termination functions for the Grasp-v1 task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def grasp_success(
    env: ManagerBasedRLEnv,
    eef_link_name: str,
    object_cfg: SceneEntityCfg,
    lift_height: float = 0.05,
    grasp_distance: float = 0.03,
    hold_duration: float = 1.0,
) -> torch.Tensor:
    """Terminate (success) when object is lifted and held for duration.

    This is a success termination - the episode ends successfully when
    the object has been stably grasped and lifted.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    is_lifted = height > lift_height

    # Check if EEF is close to object
    object_pos = obj.data.root_pos_w - env.scene.env_origins
    body_pos_w = env.scene["robot"].data.body_pos_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    eef_pos = body_pos_w[:, eef_idx] - env.scene.env_origins
    eef_dist = torch.norm(object_pos - eef_pos, dim=1)
    is_grasping = eef_dist < grasp_distance

    # Track hold duration
    attr_name = f"grasp_hold_counter_{object_cfg.name}"
    if not hasattr(env, attr_name):
        setattr(env, attr_name, torch.zeros(env.num_envs, device=env.device))

    hold_counter = getattr(env, attr_name)
    should_count = is_lifted & is_grasping
    hold_counter = torch.where(
        should_count,
        hold_counter + env.step_dt,
        torch.zeros_like(hold_counter),
    )
    setattr(env, attr_name, hold_counter)

    return hold_counter > hold_duration
