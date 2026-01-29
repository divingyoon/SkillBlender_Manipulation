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

from __future__ import annotations

import math

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply


def cup_tipped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_tilt_deg: float = 45.0,
) -> torch.Tensor:
    """Terminate if the cup tilts beyond the allowed angle from world-up.

    Args:
        env: The RL environment.
        asset_cfg: The cup asset configuration.
        max_tilt_deg: Maximum allowed tilt angle from world-up before termination.
    """
    cup = env.scene[asset_cfg.name]
    cup_quat_w = cup.data.root_quat_w
    # Cup local z-axis in world frame.
    z_axis_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(cup.data.root_pos_w)
    z_axis_world = quat_apply(cup_quat_w, z_axis_local)
    # Compare with world-up.
    world_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(cup.data.root_pos_w)
    dot = torch.sum(z_axis_world * world_up, dim=-1)
    # Tilted if angle > max_tilt_deg => dot < cos(max_tilt_deg)
    cos_thresh = math.cos(math.radians(max_tilt_deg))
    return dot < cos_thresh
