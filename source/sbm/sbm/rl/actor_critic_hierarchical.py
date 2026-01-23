# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Modifications copyright (c) 2025 Enactic, Inc.

from __future__ import annotations

import os
import re
from copy import deepcopy
import inspect

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic

try:
    from tensordict import TensorDict
except ImportError:  # pragma: no cover - tensordict may be absent in some envs
    TensorDict = None


class ActorCriticHierarchical(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        obs_context_len=1,
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        super().__init__()

        self.obs_context_len = obs_context_len
        actor_hidden_dims = actor_hidden_dims or [256, 256, 256]
        critic_hidden_dims = critic_hidden_dims or [256, 256, 256]

        activation_fn = get_activation(activation)
        self.device = kwargs.get("device")
        self.frame_stack = kwargs.get("frame_stack")
        self.command_dim = kwargs.get("command_dim")
        self.num_dofs = kwargs.get("num_dofs")
        self.command_slice = kwargs.get("command_slice")
        self.low_level_obs_groups = kwargs.get("low_level_obs_groups")
        self.obs_groups = None
        self.disable_skill_selection_until_iter = kwargs.get("disable_skill_selection_until_iter")
        self.current_learning_iteration = 0

        if TensorDict is not None and isinstance(num_actor_obs, TensorDict):
            obs = num_actor_obs
            obs_groups = num_critic_obs
            if not isinstance(obs_groups, dict):
                raise ValueError("obs_groups must be provided when observations are a TensorDict.")
            self.obs_groups = obs_groups
            num_actor_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
            num_critic_obs = sum(obs[group].shape[-1] for group in obs_groups["critic"])
            if self.device is None:
                self.device = str(obs.device)

        if self.device is None or str(self.device).lower() == "none":
            self.device = "cpu"

        if self.frame_stack is None or self.command_dim is None:
            raise ValueError("frame_stack and command_dim must be provided in policy kwargs.")
        if self.command_slice is not None and len(self.command_slice) != 2:
            raise ValueError("command_slice must be a (start, end) pair.")

        self._get_low_level_policies(kwargs)
        num_output = self._get_num_output(self.frame_stack)

        actor_layers = [nn.Linear(num_actor_obs, actor_hidden_dims[0]), activation_fn]
        for idx, hidden_dim in enumerate(actor_hidden_dims):
            is_last = idx == len(actor_hidden_dims) - 1
            next_dim = num_output if is_last else actor_hidden_dims[idx + 1]
            actor_layers.append(nn.Linear(hidden_dim, next_dim))
            if not is_last:
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), activation_fn]
        for idx, hidden_dim in enumerate(critic_hidden_dims):
            is_last = idx == len(critic_hidden_dims) - 1
            next_dim = 1 if is_last else critic_hidden_dims[idx + 1]
            critic_layers.append(nn.Linear(hidden_dim, next_dim))
            if not is_last:
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def _get_low_level_policies(self, kwargs):
        skill_dict = kwargs.get("skill_dict")
        if not skill_dict:
            raise ValueError("skill_dict is required to build hierarchical policies.")

        self.skill_names = []
        self.policy_list = []
        self.skill_command_dims = []
        self.low_high_list = []

        for name, value in skill_dict.items():
            if "command_dim" not in value:
                raise ValueError(f"Skill '{name}' must define command_dim.")

            if "policy" in value:
                policy = value["policy"]
                action_dim = self.num_dofs or policy[-1].out_features
            else:
                policy, action_dim = self._load_policy_from_checkpoint(name, value)
            policy = policy.to(self.device)

            if self.num_dofs is None:
                self.num_dofs = action_dim
            elif self.num_dofs != action_dim:
                raise ValueError(
                    f"Skill '{name}' action dim {action_dim} does not match num_dofs={self.num_dofs}."
                )

            self.skill_names.append(name)
            self.policy_list.append(policy)
            self.skill_command_dims.append(int(value["command_dim"]))
            self.low_high_list.append(value.get("low_high"))

        self.num_skills = len(self.policy_list)

    def _load_policy_from_checkpoint(self, name, skill_cfg):
        checkpoint_path = _resolve_checkpoint_path(skill_cfg)
        state_dict = _load_model_state_dict(checkpoint_path, self.device)
        actor_obs_dim, actor_hidden, action_dim = _infer_mlp_dims(state_dict, "actor")
        critic_obs_dim, critic_hidden, _ = _infer_mlp_dims(state_dict, "critic")

        policy_kwargs = deepcopy(skill_cfg.get("policy_kwargs", {}))
        policy_kwargs.setdefault("actor_hidden_dims", actor_hidden)
        policy_kwargs.setdefault("critic_hidden_dims", critic_hidden)
        policy_kwargs.setdefault("activation", skill_cfg.get("activation", "elu"))
        policy_kwargs.setdefault("init_noise_std", skill_cfg.get("init_noise_std", 1.0))
        if any(key.startswith("actor_obs_normalizer.") for key in state_dict):
            policy_kwargs.setdefault("actor_obs_normalization", True)
        if any(key.startswith("critic_obs_normalizer.") for key in state_dict):
            policy_kwargs.setdefault("critic_obs_normalization", True)
        if "log_std" in state_dict:
            policy_kwargs.setdefault("noise_std_type", "log")

        if self.device is None:
            self.device = "cpu"
        policy = _build_actor_critic(
            actor_obs_dim,
            critic_obs_dim,
            action_dim,
            policy_kwargs,
            self.device,
        )
        policy.load_state_dict(state_dict)
        _freeze_module(policy)
        return policy.actor, action_dim

    def _get_num_output(self, frame_stack=1):
        command_total = sum(self.skill_command_dims)
        num_output = command_total + (self.num_skills * self.num_dofs)
        return num_output * frame_stack

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _replace_observations(self, observations, command, low_high=None):
        if low_high is not None:
            low, high = low_high
            command = torch.clamp(command, low, high)
        new_observations = observations.clone().reshape(observations.shape[0], self.frame_stack, -1)
        num_envs, frame_stack, num_single_obs = new_observations.shape
        new_command = command.reshape(num_envs, frame_stack, -1)
        # Match SkillBlender behavior: commands occupy the leading slice.
        state_dim = num_single_obs - self.command_dim
        replaced = torch.zeros((num_envs, frame_stack, state_dim + new_command.shape[-1]), device=self.device)
        replaced[:, :, new_command.shape[-1] :] = new_observations[:, :, self.command_dim :]
        replaced[:, :, : new_command.shape[-1]] = new_command
        return replaced.reshape(num_envs, -1)

    def _resolve_command_slice(self, num_single_obs: int) -> tuple[int, int]:
        if self.command_slice is None:
            start, end = 0, self.command_dim
        else:
            start, end = int(self.command_slice[0]), int(self.command_slice[1])
        if start < 0:
            start = num_single_obs + start
        if end < 0:
            end = num_single_obs + end
        if start < 0 or end < 0 or start > num_single_obs or end > num_single_obs:
            raise ValueError(f"command_slice {self.command_slice} out of bounds for obs dim {num_single_obs}.")
        if end < start:
            raise ValueError(f"command_slice {self.command_slice} must satisfy start <= end.")
        if (end - start) != self.command_dim:
            raise ValueError(
                f"command_slice length {end - start} does not match command_dim={self.command_dim}."
            )
        return start, end

    def _actor(self, observations, low_level_obs=None, full_obs=None):
        obs_device = observations.device
        if obs_device is not None and str(obs_device) != str(self.device):
            self._move_policies_to_device(obs_device)
        raw_mean = self.actor(observations)
        input_to_low = raw_mean[:, : -(self.num_skills * self.num_dofs)]
        mask_to_low = raw_mean[:, -(self.num_skills * self.num_dofs) :]
        if not hasattr(self, "_log_counter"):
            self._log_counter = 0
        self._log_counter += 1
        masks = []
        for i in range(self.num_skills):
            mask = mask_to_low[:, i * self.num_dofs : (i + 1) * self.num_dofs]
            masks.append(mask)
        masks = torch.stack(masks, dim=1)
        masks = torch.softmax(masks, dim=1)
        if (
            self.training
            and self.disable_skill_selection_until_iter is not None
            and self.current_learning_iteration < self.disable_skill_selection_until_iter
        ):
            masks = torch.full_like(masks, 1.0 / float(self.num_skills))
        self._last_skill_masks = masks.detach()

        # Intentionally silent during inference/play to avoid log spam.


        base_obs = low_level_obs if low_level_obs is not None else observations
        means = []
        command_offset = 0
        for i in range(self.num_skills):
            curr_command_dim = self.skill_command_dims[i]
            cmd_slice = input_to_low[:, command_offset : command_offset + curr_command_dim]
            command_offset += curr_command_dim
            # Intentionally silent during inference/play to avoid log spam.
            cmd_obs = self._replace_observations(
                base_obs,
                cmd_slice,
                low_high=self.low_high_list[i],
            )
            # Intentionally silent during inference/play to avoid log spam.
            if hasattr(cmd_obs, "is_inference") and cmd_obs.is_inference():
                cmd_obs = cmd_obs.clone()
            with torch.no_grad():
                low_action = self.policy_list[i](cmd_obs)
            mean = low_action * masks[:, i]
            means.append(mean)
        actions_mean = sum(means)
        return {"actions_mean": actions_mean, "masks": masks}

    def update_distribution(self, observations):
        actor_obs = self._get_actor_obs(observations)
        low_obs = self._get_low_level_obs(observations)
        if self.obs_context_len != 1:
            actor_obs = actor_obs[..., -1, :]
            if low_obs is not None:
                low_obs = low_obs[..., -1, :]
        mean = self._actor(actor_obs, low_level_obs=low_obs, full_obs=observations)["actions_mean"]
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actor_obs = self._get_actor_obs(observations)
        low_obs = self._get_low_level_obs(observations)
        if self.obs_context_len != 1:
            actor_obs = actor_obs[..., -1, :]
            if low_obs is not None:
                low_obs = low_obs[..., -1, :]
        return self._actor(actor_obs, low_level_obs=low_obs, full_obs=observations)

    def act_inference_hrl(self, observations):
        actor_obs = self._get_actor_obs(observations)
        low_obs = self._get_low_level_obs(observations)
        if self.obs_context_len != 1:
            actor_obs = actor_obs[..., -1, :]
            if low_obs is not None:
                low_obs = low_obs[..., -1, :]
        return self._actor(actor_obs, low_level_obs=low_obs, full_obs=observations)

    def evaluate(self, critic_observations, **kwargs):
        critic_observations = self._get_critic_obs(critic_observations)
        if self.obs_context_len != 1:
            critic_observations = critic_observations[..., -1, :]
        return self.critic(critic_observations)

    def update_normalization(self, observations):
        observations = self._get_actor_obs(observations)
        if hasattr(self, "actor_obs_normalizer"):
            self.actor_obs_normalizer.update(observations)
        if hasattr(self, "critic_obs_normalizer"):
            critic_obs = self._get_critic_obs(observations)
            self.critic_obs_normalizer.update(critic_obs)

    def _move_policies_to_device(self, device):
        self.device = device
        for idx, policy in enumerate(self.policy_list):
            policy_device = next(policy.parameters()).device
            if policy_device != device:
                self.policy_list[idx] = policy.to(device)

    def _get_actor_obs(self, observations):
        if self.obs_groups is None:
            return observations
        if TensorDict is not None and isinstance(observations, TensorDict):
            obs_list = [observations[group] for group in self.obs_groups["policy"]]
            return torch.cat(obs_list, dim=-1)
        return observations

    def _get_critic_obs(self, observations):
        if self.obs_groups is None:
            return observations
        if TensorDict is not None and isinstance(observations, TensorDict):
            obs_list = [observations[group] for group in self.obs_groups["critic"]]
            return torch.cat(obs_list, dim=-1)
        return observations

    def _get_low_level_obs(self, observations):
        if self.low_level_obs_groups is None:
            return self._get_actor_obs(observations)
        if TensorDict is not None and isinstance(observations, TensorDict):
            obs_list = [observations[group] for group in self.low_level_obs_groups]
            return torch.cat(obs_list, dim=-1)
        return observations

    def _get_skill_bias(self, observations):
        if TensorDict is None or observations is None or not isinstance(observations, TensorDict):
            return None
        if "high_level" not in observations:
            return None
        high_level = observations["high_level"]
        if high_level.shape[-1] < 2:
            return None
        # Use the last two dims as [reach, grasp2g] bias.
        bias = high_level[..., -2:]
        # Map bias to skill order.
        skill_bias = torch.zeros((bias.shape[0], self.num_skills), device=bias.device)
        for idx, name in enumerate(self.skill_names):
            if "reach" in name:
                skill_bias[:, idx] = bias[:, 0]
            elif "grasp_2g" in name or "grasp2g" in name:
                skill_bias[:, idx] = bias[:, 1]
        return skill_bias


class ActorCriticHierarchicalDualHead(ActorCriticHierarchical):
    """Hierarchical actor with shared encoder and per-side output heads."""

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        obs_context_len=1,
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if TensorDict is not None and isinstance(num_actor_obs, TensorDict):
            obs_groups = num_critic_obs
            if not isinstance(obs_groups, dict):
                raise ValueError("obs_groups must be provided when observations are a TensorDict.")
            self._actor_input_dim = sum(num_actor_obs[group].shape[-1] for group in obs_groups["policy"])
        else:
            self._actor_input_dim = num_actor_obs
        self._actor_hidden_dims = actor_hidden_dims or [256, 256, 256]
        self._actor_activation = activation
        super().__init__(
            num_actor_obs,
            num_critic_obs,
            num_actions,
            obs_context_len=obs_context_len,
            actor_hidden_dims=self._actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            **kwargs,
        )
        self.command_split_index = kwargs.get("command_split_index")
        self.dof_split_index = kwargs.get("dof_split_index")
        self._build_dual_head_actor()

    def _build_dual_head_actor(self):
        activation_fn = get_activation(self._actor_activation)
        # Keep a lightweight, parameter-free actor stub so exporters can query in_features.
        self.actor = _ActorInFeaturesStub(self._actor_input_dim)
        left_cmd_dims = []
        right_cmd_dims = []
        for dim in self.skill_command_dims:
            split = dim // 2 if self.command_split_index is None else int(self.command_split_index)
            split = max(0, min(split, dim))
            left_cmd_dims.append(split)
            right_cmd_dims.append(dim - split)
        self._left_skill_command_dims = left_cmd_dims
        self._right_skill_command_dims = right_cmd_dims

        if self.num_dofs is None:
            raise ValueError("num_dofs must be available before building dual-head actor.")
        dof_split = self.num_dofs // 2 if self.dof_split_index is None else int(self.dof_split_index)
        dof_split = max(0, min(dof_split, self.num_dofs))
        self._left_dofs = dof_split
        self._right_dofs = self.num_dofs - dof_split

        self._left_command_dim = sum(self._left_skill_command_dims)
        self._right_command_dim = sum(self._right_skill_command_dims)
        self._per_frame_left_dim = self._left_command_dim + self.num_skills * self._left_dofs
        self._per_frame_right_dim = self._right_command_dim + self.num_skills * self._right_dofs

        encoder_layers = []
        prev_dim = self._actor_input_dim
        for hidden_dim in self._actor_hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(activation_fn)
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.head_left = nn.Linear(prev_dim, self._per_frame_left_dim * self.frame_stack)
        self.head_right = nn.Linear(prev_dim, self._per_frame_right_dim * self.frame_stack)

    def _merge_dual_frame(self, left_frame: torch.Tensor, right_frame: torch.Tensor) -> torch.Tensor:
        batch = left_frame.shape[0]
        left_cmd = left_frame[:, : self._left_command_dim] if self._left_command_dim > 0 else None
        left_mask = left_frame[:, self._left_command_dim :] if self._per_frame_left_dim > self._left_command_dim else None
        right_cmd = right_frame[:, : self._right_command_dim] if self._right_command_dim > 0 else None
        right_mask = (
            right_frame[:, self._right_command_dim :]
            if self._per_frame_right_dim > self._right_command_dim
            else None
        )

        cmd_parts = []
        left_offset = 0
        right_offset = 0
        for left_dim, right_dim in zip(self._left_skill_command_dims, self._right_skill_command_dims):
            left_slice = (
                left_cmd[:, left_offset : left_offset + left_dim] if left_dim > 0 else None
            )
            right_slice = (
                right_cmd[:, right_offset : right_offset + right_dim] if right_dim > 0 else None
            )
            left_offset += left_dim
            right_offset += right_dim
            if left_slice is None:
                cmd_parts.append(right_slice)
            elif right_slice is None:
                cmd_parts.append(left_slice)
            else:
                cmd_parts.append(torch.cat((left_slice, right_slice), dim=1))
        commands = torch.cat(cmd_parts, dim=1) if cmd_parts else torch.zeros((batch, 0), device=left_frame.device)

        if self.num_skills == 0 or self.num_dofs == 0:
            masks = torch.zeros((batch, 0), device=left_frame.device)
        else:
            if self._left_dofs > 0:
                left_mask = left_mask.view(batch, self.num_skills, self._left_dofs)
            else:
                left_mask = torch.zeros((batch, self.num_skills, 0), device=left_frame.device)
            if self._right_dofs > 0:
                right_mask = right_mask.view(batch, self.num_skills, self._right_dofs)
            else:
                right_mask = torch.zeros((batch, self.num_skills, 0), device=left_frame.device)

            masks_per_skill = []
            for idx in range(self.num_skills):
                masks_per_skill.append(torch.cat((left_mask[:, idx, :], right_mask[:, idx, :]), dim=1))
            masks = torch.cat(masks_per_skill, dim=1)

        return torch.cat((commands, masks), dim=1)

    def _dual_raw_mean(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.encoder(observations)
        raw_left = self.head_left(features)
        raw_right = self.head_right(features)
        if self.frame_stack == 1:
            return self._merge_dual_frame(raw_left, raw_right)
        batch = raw_left.shape[0]
        left = raw_left.view(batch, self.frame_stack, self._per_frame_left_dim)
        right = raw_right.view(batch, self.frame_stack, self._per_frame_right_dim)
        merged = [self._merge_dual_frame(left[:, idx, :], right[:, idx, :]) for idx in range(self.frame_stack)]
        return torch.stack(merged, dim=1).reshape(batch, -1)

    def _actor(self, observations, low_level_obs=None, full_obs=None):
        obs_device = observations.device
        if obs_device is not None and str(obs_device) != str(self.device):
            self._move_policies_to_device(obs_device)
        raw_mean = self._dual_raw_mean(observations)
        input_to_low = raw_mean[:, : -(self.num_skills * self.num_dofs)]
        mask_to_low = raw_mean[:, -(self.num_skills * self.num_dofs) :]
        if not hasattr(self, "_log_counter"):
            self._log_counter = 0
        self._log_counter += 1
        masks = []
        for i in range(self.num_skills):
            mask = mask_to_low[:, i * self.num_dofs : (i + 1) * self.num_dofs]
            masks.append(mask)
        masks = torch.stack(masks, dim=1)
        masks = torch.softmax(masks, dim=1)
        if (
            self.training
            and self.disable_skill_selection_until_iter is not None
            and self.current_learning_iteration < self.disable_skill_selection_until_iter
        ):
            masks = torch.full_like(masks, 1.0 / float(self.num_skills))
        self._last_skill_masks = masks.detach()

        base_obs = low_level_obs if low_level_obs is not None else observations
        means = []
        command_offset = 0
        for i in range(self.num_skills):
            curr_command_dim = self.skill_command_dims[i]
            cmd_slice = input_to_low[:, command_offset : command_offset + curr_command_dim]
            command_offset += curr_command_dim
            cmd_obs = self._replace_observations(
                base_obs,
                cmd_slice,
                low_high=self.low_high_list[i],
            )
            if hasattr(cmd_obs, "is_inference") and cmd_obs.is_inference():
                cmd_obs = cmd_obs.clone()
            with torch.no_grad():
                low_action = self.policy_list[i](cmd_obs)
            mean = low_action * masks[:, i]
            means.append(mean)
        actions_mean = sum(means)
        return {"actions_mean": actions_mean, "masks": masks}


def _freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


class _ActorInFeaturesStub(nn.Module):
    """Parameter-free stub that mimics actor[0].in_features for exporters."""

    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = int(in_features)

    def __getitem__(self, idx: int):
        if idx != 0:
            raise IndexError("Actor stub only supports index 0.")
        return self

    def forward(self, x):
        return x


def _resolve_checkpoint_path(skill_cfg: dict) -> str:
    if "checkpoint_path" in skill_cfg:
        return skill_cfg["checkpoint_path"]

    experiment_name = skill_cfg.get("experiment_name")
    if experiment_name is None:
        raise ValueError("Skill config must include checkpoint_path or experiment_name.")

    log_root = skill_cfg.get("log_root")
    if log_root is None:
        sbm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        log_root = os.path.join(sbm_root, "log", "rsl_rl", experiment_name)
    log_root = os.path.expanduser(log_root)
    if not os.path.isabs(log_root):
        sbm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        log_root = os.path.join(sbm_root, log_root)

    run_dir = _select_run_dir(log_root, skill_cfg.get("load_run", "latest"))
    return _select_checkpoint(run_dir, skill_cfg.get("checkpoint", "latest"))


def _select_run_dir(log_root: str, load_run):
    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"Log root not found: {log_root}")
    runs = [d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))]
    if not runs:
        raise FileNotFoundError(f"No runs found in log root: {log_root}")

    if load_run in (None, -1, "latest", ".*"):
        return os.path.join(log_root, sorted(runs)[-1])

    if isinstance(load_run, str) and load_run in runs:
        return os.path.join(log_root, load_run)

    pattern = re.compile(str(load_run))
    matches = sorted([d for d in runs if pattern.search(d)])
    if not matches:
        raise FileNotFoundError(f"No runs match '{load_run}' in {log_root}")
    return os.path.join(log_root, matches[-1])


def _select_checkpoint(run_dir: str, checkpoint):
    if checkpoint in (None, -1, "latest", ".*"):
        return _latest_checkpoint(run_dir)

    if isinstance(checkpoint, int):
        candidate = os.path.join(run_dir, f"model_{checkpoint}.pt")
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")

    if isinstance(checkpoint, str):
        if checkpoint.endswith(".pt"):
            candidate = checkpoint if os.path.isabs(checkpoint) else os.path.join(run_dir, checkpoint)
            if os.path.exists(candidate):
                return candidate
        pattern = re.compile(checkpoint)
        matches = sorted([f for f in os.listdir(run_dir) if pattern.search(f)])
        if matches:
            return os.path.join(run_dir, matches[-1])

    raise FileNotFoundError(f"Checkpoint not found in {run_dir}: {checkpoint}")


def _latest_checkpoint(run_dir: str) -> str:
    candidates = [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    def sort_key(name: str):
        match = re.search(r"model_(\\d+)\\.pt", name)
        return int(match.group(1)) if match else -1

    latest = max(candidates, key=sort_key)
    return os.path.join(run_dir, latest)


def _load_model_state_dict(checkpoint_path: str, device):
    try:
        loaded = torch.load(checkpoint_path, map_location=device)
    except Exception:
        loaded = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in loaded:
        raise KeyError(f"model_state_dict missing in checkpoint: {checkpoint_path}")
    return loaded["model_state_dict"]


def _infer_mlp_dims(state_dict: dict, prefix: str):
    weights = []
    for key, tensor in state_dict.items():
        if key.startswith(f"{prefix}.") and key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) < 3:
                continue
            try:
                layer_idx = int(parts[1])
            except ValueError:
                continue
            weights.append((layer_idx, tensor))
    if not weights:
        raise KeyError(f"No weights found for prefix '{prefix}' in state_dict.")
    weights = [tensor for _, tensor in sorted(weights, key=lambda item: item[0])]
    input_dim = weights[0].shape[1]
    hidden_dims = [w.shape[0] for w in weights[:-1]]
    output_dim = weights[-1].shape[0]
    return input_dim, hidden_dims, output_dim


def _build_actor_critic(
    actor_obs_dim: int,
    critic_obs_dim: int,
    action_dim: int,
    policy_kwargs: dict,
    device: str,
):
    sig = inspect.signature(ActorCritic.__init__)
    params = list(sig.parameters.values())
    if len(params) > 1 and params[1].name == "obs":
        from tensordict import TensorDict

        obs = TensorDict(
            {
                "policy": torch.zeros(1, actor_obs_dim),
                "critic": torch.zeros(1, critic_obs_dim),
            },
            batch_size=[1],
        )
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        model = ActorCritic(obs, obs_groups, action_dim, **policy_kwargs)
    else:
        model = ActorCritic(actor_obs_dim, critic_obs_dim, action_dim, obs_context_len=1, **policy_kwargs)

    if device is not None and str(device).lower() != "none":
        return model.to(device)
    return model


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    if act_name == "selu":
        return nn.SELU()
    if act_name == "relu":
        return nn.ReLU()
    if act_name == "crelu":
        return nn.ReLU()
    if act_name == "lrelu":
        return nn.LeakyReLU()
    if act_name == "tanh":
        return nn.Tanh()
    if act_name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {act_name}")
