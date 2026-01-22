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

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg


@configclass
class SbmHierarchicalActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-critic config with extra fields required for hierarchical skill blending."""

    class_name: str = "ActorCriticHierarchical"
    skill_dict: dict = MISSING
    frame_stack: int = MISSING
    command_dim: int = MISSING
    command_slice: list[int] | None = None
    command_split_index: int | None = None
    dof_split_index: int | None = None
    num_dofs: int | None = None
    low_level_obs_groups: list[str] | None = None
    disable_skill_selection_until_iter: int | None = None
