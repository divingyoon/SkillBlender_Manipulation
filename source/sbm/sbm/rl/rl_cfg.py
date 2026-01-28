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


@configclass
class SbmDualHeadActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-critic config for shared-encoder dual-head low-level PPO.

    양손 비대칭 학습 문제 해결을 위한 옵션:
    - separate_noise_std: 좌/우 독립적 noise std (방법3)
    - dual_critic: 좌/우 분리된 critic (방법5)
    """

    class_name: str = "ActorCriticDualHead"
    dof_split_index: int | None = None

    # [방법3] 좌/우 독립적 noise std 사용 여부
    # True: 좌/우 팔이 독립적으로 탐험 (한쪽 성공이 다른쪽에 영향 안줌)
    # False: 기존 방식 (전체 공유 noise std)
    separate_noise_std: bool = True

    # [방법5] 좌/우 분리된 critic 사용 여부
    # True: 좌/우 각각의 value function 학습 (독립적 value 추정)
    # False: 기존 방식 (단일 critic)
    dual_critic: bool = True
