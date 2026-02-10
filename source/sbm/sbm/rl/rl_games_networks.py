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

"""Dual-head network implementations for rl_games.

This module provides custom network architectures for rl_games that support
separate encoders for left and right arms in bimanual manipulation tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import NetworkBuilder, A2CBuilder
from rl_games.algos_torch.models import ModelA2CContinuousLogStd


def _build_mlp(input_dim: int, hidden_dims: list[int], activation: str) -> nn.Sequential:
    """Build an MLP encoder."""
    layers = []
    prev = input_dim
    act_fn = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "selu": nn.SELU,
        "leaky_relu": nn.LeakyReLU,
    }.get(activation.lower(), nn.ELU)

    for dim in hidden_dims:
        layers.append(nn.Linear(prev, dim))
        layers.append(act_fn())
        prev = dim
    return nn.Sequential(*layers)


class DualHeadA2CNetwork(NetworkBuilder.BaseNetwork):
    """Dual-head A2C network with separate encoders for left/right arms.

    This network splits observations and actions between left and right arms,
    allowing each arm to have its own encoder. This is useful for:
    - Rollout-based training where pre-trained policies are executed
    - Preventing distribution shift when one arm's policy is frozen
    - Bimanual manipulation tasks requiring arm-specific representations
    """

    def __init__(self, params, **kwargs):
        super().__init__()

        actions_num = kwargs.pop("actions_num")
        input_shape = kwargs.pop("input_shape")
        self.value_size = kwargs.pop("value_size", 1)

        # Extract dual-head specific parameters
        self.dof_split_index = params.get("dof_split_index", actions_num // 2)
        self.obs_split_index = params.get("obs_split_index", input_shape[0] // 2)
        self.separate_encoders = params.get("separate_encoders", True)
        self.separate_noise_std = params.get("separate_noise_std", True)
        self.dual_critic = params.get("dual_critic", True)

        # Network configuration
        mlp_config = params.get("mlp", {})
        hidden_dims = mlp_config.get("units", [256, 128, 64])
        activation = mlp_config.get("activation", "elu")

        # Action dimensions
        self.left_actions = int(self.dof_split_index)
        self.right_actions = int(actions_num - self.dof_split_index)

        # Observation dimensions
        input_dim = input_shape[0]
        if self.separate_encoders and 0 < self.obs_split_index < input_dim:
            self.left_obs_dim = int(self.obs_split_index)
            self.right_obs_dim = int(input_dim - self.obs_split_index)
        else:
            self.left_obs_dim = input_dim
            self.right_obs_dim = input_dim

        # Build actor encoders
        if self.separate_encoders:
            self.actor_encoder_left = _build_mlp(self.left_obs_dim, hidden_dims, activation)
            self.actor_encoder_right = _build_mlp(self.right_obs_dim, hidden_dims, activation)
        else:
            self.actor_encoder = _build_mlp(input_dim, hidden_dims, activation)

        # Actor heads
        encoder_out_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.mu_left = nn.Linear(encoder_out_dim, self.left_actions)
        self.mu_right = nn.Linear(encoder_out_dim, self.right_actions)

        # Noise std (learnable)
        if self.separate_noise_std:
            self.log_std_left = nn.Parameter(torch.zeros(self.left_actions))
            self.log_std_right = nn.Parameter(torch.zeros(self.right_actions))
        else:
            self.log_std = nn.Parameter(torch.zeros(actions_num))

        # Build critic
        if self.dual_critic:
            if self.separate_encoders:
                self.critic_encoder_left = _build_mlp(self.left_obs_dim, hidden_dims, activation)
                self.critic_encoder_right = _build_mlp(self.right_obs_dim, hidden_dims, activation)
            else:
                self.critic_encoder = _build_mlp(input_dim, hidden_dims, activation)
            self.value_left = nn.Linear(encoder_out_dim, self.value_size)
            self.value_right = nn.Linear(encoder_out_dim, self.value_size)
        else:
            self.critic_encoder = _build_mlp(input_dim, hidden_dims, activation)
            self.value = nn.Linear(encoder_out_dim, self.value_size)

        # Initialize weights
        self._init_weights()

        print(f"[DualHeadA2CNetwork] dof_split={self.dof_split_index}, obs_split={self.obs_split_index}")
        print(f"  Actor: left_actions={self.left_actions}, right_actions={self.right_actions}")
        print(f"  Separate encoders: {self.separate_encoders}, Dual critic: {self.dual_critic}")

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, obs_dict):
        obs = obs_dict["obs"]

        # Split observations if using separate encoders
        if self.separate_encoders and 0 < self.obs_split_index < obs.shape[-1]:
            obs_left = obs[..., :self.obs_split_index]
            obs_right = obs[..., self.obs_split_index:]
        else:
            obs_left = obs
            obs_right = obs

        # Actor forward
        if self.separate_encoders:
            feat_left = self.actor_encoder_left(obs_left)
            feat_right = self.actor_encoder_right(obs_right)
        else:
            feat = self.actor_encoder(obs)
            feat_left = feat
            feat_right = feat

        mu_left = self.mu_left(feat_left)
        mu_right = self.mu_right(feat_right)
        mu = torch.cat([mu_left, mu_right], dim=-1)

        # Log std
        if self.separate_noise_std:
            log_std = torch.cat([self.log_std_left, self.log_std_right], dim=-1)
        else:
            log_std = self.log_std
        sigma = log_std.exp().unsqueeze(0).expand_as(mu)

        # Critic forward
        if self.dual_critic:
            if self.separate_encoders:
                critic_feat_left = self.critic_encoder_left(obs_left)
                critic_feat_right = self.critic_encoder_right(obs_right)
            else:
                critic_feat = self.critic_encoder(obs)
                critic_feat_left = critic_feat
                critic_feat_right = critic_feat
            value_left = self.value_left(critic_feat_left)
            value_right = self.value_right(critic_feat_right)
            value = (value_left + value_right) / 2.0
        else:
            critic_feat = self.critic_encoder(obs)
            value = self.value(critic_feat)

        return mu, mu * 0.0 + sigma, value, None

    def is_rnn(self):
        return False

    def get_default_rnn_state(self):
        return None


class DualHeadA2CBuilder(NetworkBuilder):
    """Network builder for DualHeadA2CNetwork."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = None

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return DualHeadA2CNetwork(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)


class DualHeadModelA2CContinuousLogStd(ModelA2CContinuousLogStd):
    """Model wrapper for DualHeadA2CNetwork compatible with rl_games."""

    def __init__(self, network):
        super().__init__(network)

    def build(self, config):
        # Get network configuration
        net = self.network_builder.build("dualhead_a2c", **config)
        return DualHeadModelA2CContinuousLogStd.Network(net)

    class Network(ModelA2CContinuousLogStd.Network):
        """Inner network class."""

        def __init__(self, a2c_network):
            super().__init__(a2c_network)


def register_rl_games_dualhead():
    """Register dual-head network with rl_games."""
    from rl_games.algos_torch import model_builder

    # Register the network builder
    model_builder.register_network("dualhead_a2c", DualHeadA2CBuilder)

    print("[sbm.rl] Registered rl_games dual-head network: dualhead_a2c")
