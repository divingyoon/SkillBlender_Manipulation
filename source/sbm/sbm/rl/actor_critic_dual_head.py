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

from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP
from rsl_rl.utils import resolve_nn_activation


class _DualHeadActor(nn.Module):
    """Shared encoder with per-side linear heads."""

    def __init__(self, input_dim: int, hidden_dims: list[int], left_dim: int, right_dim: int, activation: str) -> None:
        super().__init__()
        act = resolve_nn_activation(activation)
        layers: list[nn.Module] = []
        prev = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(act)
            prev = dim
        self.encoder = nn.Sequential(*layers)
        self.head_left = nn.Linear(prev, left_dim)
        self.head_right = nn.Linear(prev, right_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        left = self.head_left(feat)
        right = self.head_right(feat)
        return torch.cat((left, right), dim=-1)


class ActorCriticDualHead(nn.Module):
    """Low-level PPO actor-critic with shared encoder and dual action heads.

    양손 비대칭 학습 문제 해결을 위한 기능 포함:
    - [방법3] separate_noise_std: 좌/우 독립적 noise std로 독립적 탐험
    - [방법5] dual_critic: 좌/우 분리된 critic으로 독립적 value 추정
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        dof_split_index: int | None = None,
        # [방법3] 좌/우 독립적 noise std 사용 여부
        separate_noise_std: bool = True,
        # [방법5] 좌/우 분리된 critic 사용 여부
        dual_critic: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticDualHead.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        if state_dependent_std:
            raise ValueError("ActorCriticDualHead does not support state-dependent std.")

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        # Determine action split
        split = num_actions // 2 if dof_split_index is None else int(dof_split_index)
        split = max(0, min(split, num_actions))
        self._left_actions = split
        self._right_actions = num_actions - split

        # Actor (shared encoder + dual heads)
        actor_hidden = list(actor_hidden_dims) if isinstance(actor_hidden_dims, (list, tuple)) else [actor_hidden_dims]
        self.actor = _DualHeadActor(num_actor_obs, actor_hidden, self._left_actions, self._right_actions, activation)
        print(f"Actor DualHead: {self.actor}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # [방법5] Dual Critic: 좌/우 분리된 critic으로 독립적 value 추정
        # 양손의 value 추정이 섞이는 것을 방지
        self.dual_critic = dual_critic
        if dual_critic:
            self.critic_left = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
            self.critic_right = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
            print(f"Critic DualHead (Left): {self.critic_left}")
            print(f"Critic DualHead (Right): {self.critic_right}")
        else:
            self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
            print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # [방법3] Action noise: 좌/우 독립적 noise std로 독립적 탐험 유도
        # 한쪽이 성공해도 다른쪽이 독립적으로 탐험할 수 있음
        self.noise_std_type = noise_std_type
        self.separate_noise_std = separate_noise_std

        if separate_noise_std:
            # 좌/우 독립적 noise std
            if self.noise_std_type == "scalar":
                self.std_left = nn.Parameter(init_noise_std * torch.ones(self._left_actions))
                self.std_right = nn.Parameter(init_noise_std * torch.ones(self._right_actions))
            elif self.noise_std_type == "log":
                self.log_std_left = nn.Parameter(torch.log(init_noise_std * torch.ones(self._left_actions)))
                self.log_std_right = nn.Parameter(torch.log(init_noise_std * torch.ones(self._right_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            print(f"[방법3] Separate Noise Std enabled: left={self._left_actions}, right={self._right_actions}")
        else:
            # 기존 방식: 전체 공유 noise std
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: TensorDict) -> None:
        mean = self.actor(obs)

        # [방법3] 좌/우 독립적 noise std 처리
        if self.separate_noise_std:
            if self.noise_std_type == "scalar":
                std_left = self.std_left.expand(mean.shape[0], -1)
                std_right = self.std_right.expand(mean.shape[0], -1)
            elif self.noise_std_type == "log":
                std_left = torch.exp(self.log_std_left).expand(mean.shape[0], -1)
                std_right = torch.exp(self.log_std_right).expand(mean.shape[0], -1)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
            std = torch.cat([std_left, std_right], dim=-1)
        else:
            # 기존 방식: 전체 공유 noise std
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")

        std = torch.clamp(std, min=1e-6)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self.actor(obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)

        # [방법5] Dual Critic: 좌/우 critic의 평균값 반환
        if self.dual_critic:
            value_left = self.critic_left(obs)
            value_right = self.critic_right(obs)
            return (value_left + value_right) / 2.0
        else:
            return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)
