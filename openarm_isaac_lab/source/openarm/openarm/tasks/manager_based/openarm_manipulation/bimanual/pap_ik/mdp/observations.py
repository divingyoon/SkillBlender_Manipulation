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

"""Observation helpers missing from upstream MDP."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def root_state(
    env: ManagerBasedEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root state (pos, quat, lin vel, ang vel) in the environment frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    state = asset.data.root_state_w.clone()
    state[:, :3] -= env.scene.env_origins
    if make_quat_unique:
        state[:, 3:7] = math_utils.quat_unique(state[:, 3:7])
    return state


def body_state(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Body state (pose + velocity) in the environment frame for selected bodies."""
    asset: Articulation = env.scene[asset_cfg.name]
    state = asset.data.body_state_w[:, asset_cfg.body_ids, :].clone()
    state[..., :3] -= env.scene.env_origins.unsqueeze(1)
    return state.reshape(env.num_envs, -1)
