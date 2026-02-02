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

from typing import Sequence

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


def _resolve_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "elu":
        return nn.ELU()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "leaky_relu":
        return nn.LeakyReLU()
    if key == "selu":
        return nn.SELU()
    raise ValueError(f"Unsupported activation: {name}")


def _build_mlp(input_dim: int, hidden_dims: Sequence[int], activation: str) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    prev = int(input_dim)
    act = _resolve_activation(activation)
    for dim in hidden_dims:
        layers.append(nn.Linear(prev, int(dim)))
        layers.append(act)
        prev = int(dim)
    return nn.Sequential(*layers), prev


class DualHeadGaussianModel(Model, GaussianMixin):
    """Shared-encoder dual-head Gaussian policy for skrl PPO."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
        actor_hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        dof_split_index: int | None = None,
        actor_obs_split_index: int | None = None,
        separate_actor_encoders: bool = False,
        separate_noise_std: bool = True,
        **kwargs,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, initial_log_std)

        if kwargs:
            _ = kwargs  # Unused extra kwargs from config; keep for forward-compat.

        split = self.num_actions // 2 if dof_split_index is None else int(dof_split_index)
        split = max(0, min(split, self.num_actions))
        self._left_actions = split
        self._right_actions = self.num_actions - split

        self._actor_obs_split_index = actor_obs_split_index
        self._separate_actor_encoders = separate_actor_encoders
        if separate_actor_encoders:
            left_in = self.num_observations
            right_in = self.num_observations
            if actor_obs_split_index is not None and 0 < int(actor_obs_split_index) < int(self.num_observations):
                left_in = int(actor_obs_split_index)
                right_in = int(self.num_observations - int(actor_obs_split_index))
            self.encoder_left, left_dim = _build_mlp(left_in, actor_hidden_dims, activation)
            self.encoder_right, right_dim = _build_mlp(right_in, actor_hidden_dims, activation)
            self.head_left = nn.Linear(left_dim, self._left_actions)
            self.head_right = nn.Linear(right_dim, self._right_actions)
        else:
            self.encoder, last_dim = _build_mlp(self.num_observations, actor_hidden_dims, activation)
            self.head_left = nn.Linear(last_dim, self._left_actions)
            self.head_right = nn.Linear(last_dim, self._right_actions)

        self.separate_noise_std = separate_noise_std
        self._clip_log_std = clip_log_std
        self._min_log_std = float(min_log_std)
        self._max_log_std = float(max_log_std)

        if separate_noise_std:
            self.log_std_left = nn.Parameter(torch.full((self._left_actions,), float(initial_log_std)))
            self.log_std_right = nn.Parameter(torch.full((self._right_actions,), float(initial_log_std)))
        else:
            self.log_std_shared = nn.Parameter(torch.full((1,), float(initial_log_std)))

    def compute(self, inputs, role):
        states = inputs["states"]
        if self._separate_actor_encoders and self._actor_obs_split_index is not None:
            if 0 < int(self._actor_obs_split_index) < states.shape[-1]:
                left_states = states[:, : int(self._actor_obs_split_index)]
                right_states = states[:, int(self._actor_obs_split_index) :]
            else:
                left_states = states
                right_states = states
            feat_left = self.encoder_left(left_states)
            feat_right = self.encoder_right(right_states)
            mean_left = self.head_left(feat_left)
            mean_right = self.head_right(feat_right)
        elif self._separate_actor_encoders:
            feat_left = self.encoder_left(states)
            feat_right = self.encoder_right(states)
            mean_left = self.head_left(feat_left)
            mean_right = self.head_right(feat_right)
        else:
            features = self.encoder(states)
            mean_left = self.head_left(features)
            mean_right = self.head_right(features)
        mean = torch.cat((mean_left, mean_right), dim=-1)

        if self.separate_noise_std:
            log_std = torch.cat((self.log_std_left, self.log_std_right), dim=-1)
        else:
            log_std = self.log_std_shared.expand(self.num_actions)

        if self._clip_log_std:
            log_std = torch.clamp(log_std, self._min_log_std, self._max_log_std)

        log_std = log_std.expand_as(mean)
        return mean, log_std, {}


class DualHeadValueModel(Model, DeterministicMixin):
    """Dual-critic value model for skrl PPO."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        dual_critic: bool = True,
        critic_obs_split_index: int | None = None,
        separate_critic_encoders: bool = False,
        **kwargs,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        if kwargs:
            _ = kwargs

        self.dual_critic = dual_critic
        self._critic_obs_split_index = critic_obs_split_index
        self._separate_critic_encoders = separate_critic_encoders
        if dual_critic:
            left_in = self.num_observations
            right_in = self.num_observations
            if separate_critic_encoders and critic_obs_split_index is not None:
                if 0 < int(critic_obs_split_index) < int(self.num_observations):
                    left_in = int(critic_obs_split_index)
                    right_in = int(self.num_observations - int(critic_obs_split_index))
            self.critic_left, left_dim = _build_mlp(left_in, critic_hidden_dims, activation)
            self.critic_right, right_dim = _build_mlp(right_in, critic_hidden_dims, activation)
            self.value_left = nn.Linear(left_dim, 1)
            self.value_right = nn.Linear(right_dim, 1)
        else:
            self.critic, last_dim = _build_mlp(self.num_observations, critic_hidden_dims, activation)
            self.value = nn.Linear(last_dim, 1)

    def compute(self, inputs, role):
        states = inputs["states"]
        if self.dual_critic:
            if self._separate_critic_encoders and self._critic_obs_split_index is not None:
                if 0 < int(self._critic_obs_split_index) < states.shape[-1]:
                    left_states = states[:, : int(self._critic_obs_split_index)]
                    right_states = states[:, int(self._critic_obs_split_index) :]
                else:
                    left_states = states
                    right_states = states
            else:
                left_states = states
                right_states = states
            left_feat = self.critic_left(left_states)
            right_feat = self.critic_right(right_states)
            value_left = self.value_left(left_feat)
            value_right = self.value_right(right_feat)
            value = (value_left + value_right) / 2.0
        else:
            feat = self.critic(states)
            value = self.value(feat)
        return value, {}
