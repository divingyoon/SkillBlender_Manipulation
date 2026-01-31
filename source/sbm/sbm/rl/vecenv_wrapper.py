# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class RslRlVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for the RSL-RL library.

    This local copy lets downstream projects extend swap pairs without
    patching IsaacLab's internal wrapper.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv | DirectRLEnv,
        clip_actions: float | None = None,
        swap_lr: bool = False,
        swap_prob: float = 0.5,
        swap_obs_term_pairs: list[tuple[str, str]] | None = None,
        swap_action_term_pairs: list[tuple[str, str]] | None = None,
    ):
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )

        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions
        self._swap_lr = swap_lr
        self._swap_prob = float(swap_prob)

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        # modify the action space to the clip range
        self._modify_action_space()

        # prepare swap helpers (if enabled)
        self._swap_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._obs_term_slices: dict[str, dict[str, slice]] = {}
        self._obs_group_concat: dict[str, bool] = {}
        self._action_term_slices: dict[str, slice] = {}
        self._reward_term_pairs: list[tuple[str, str]] = []
        self._reward_term_indices: list[tuple[int, int]] = []

        # If not provided, infer swap pairs from observation term names at runtime.
        self._swap_obs_term_pairs = swap_obs_term_pairs or []
        self._swap_action_term_pairs = swap_action_term_pairs or [
            ("left_arm_action", "right_arm_action"),
            ("left_hand_action", "right_hand_action"),
        ]

        if self._swap_lr:
            self._build_swap_helpers()
            if not self._swap_obs_term_pairs:
                self._swap_obs_term_pairs = self._infer_obs_swap_pairs()
            self._sample_swap_mask()

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    @property
    def cfg(self) -> object:
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:  # noqa: D102
        obs_dict, extras = self.env.reset()
        if self._swap_lr:
            self._sample_swap_mask()
            self._swap_obs_inplace(obs_dict, self._swap_mask)
        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

    def get_observations(self) -> TensorDict:
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        if self._swap_lr:
            actions = self._swap_actions_inplace(actions, self._swap_mask)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if self._swap_lr:
            self._swap_reward_terms_inplace(self._swap_mask)
            if torch.any(dones.bool()):
                self._sample_swap_mask(dones.bool())
            self._swap_obs_inplace(obs_dict, self._swap_mask)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    def _build_swap_helpers(self):
        if not hasattr(self.unwrapped, "observation_manager"):
            return

        obs_mgr = self.unwrapped.observation_manager
        for group_name, term_names in obs_mgr.active_terms.items():
            self._obs_group_concat[group_name] = obs_mgr.group_obs_concatenate[group_name]
            if not obs_mgr.group_obs_concatenate[group_name]:
                continue
            term_dims = obs_mgr.group_obs_term_dim[group_name]
            term_slices: dict[str, slice] = {}
            idx = 0
            for name, dims in zip(term_names, term_dims):
                length = int(torch.prod(torch.tensor(dims)).item())
                term_slices[name] = slice(idx, idx + length)
                idx += length
            self._obs_term_slices[group_name] = term_slices

        if hasattr(self.unwrapped, "action_manager"):
            names = self.unwrapped.action_manager.active_terms
            dims = self.unwrapped.action_manager.action_term_dim
            idx = 0
            for name, dim in zip(names, dims):
                self._action_term_slices[name] = slice(idx, idx + int(dim))
                idx += int(dim)

        if hasattr(self.unwrapped, "reward_manager"):
            reward_terms = set(self.unwrapped.reward_manager.active_terms)
            for name in reward_terms:
                if "left_" in name:
                    counterpart = name.replace("left_", "right_", 1)
                    if counterpart in reward_terms:
                        self._reward_term_pairs.append((name, counterpart))
            if self._reward_term_pairs:
                name_to_idx = {n: i for i, n in enumerate(self.unwrapped.reward_manager.active_terms)}
                for left, right in self._reward_term_pairs:
                    self._reward_term_indices.append((name_to_idx[left], name_to_idx[right]))

    def _infer_obs_swap_pairs(self) -> list[tuple[str, str]]:
        if not hasattr(self.unwrapped, "observation_manager"):
            return []
        obs_mgr = self.unwrapped.observation_manager
        all_terms = set()
        for _, term_names in obs_mgr.active_terms.items():
            all_terms.update(term_names)

        pairs = set()
        for name in list(all_terms):
            if "left" in name:
                cand = name.replace("left", "right")
                if cand in all_terms:
                    pairs.add((name, cand))
            if "right" in name:
                cand = name.replace("right", "left")
                if cand in all_terms:
                    pairs.add((cand, name))
        for name in list(all_terms):
            if name.endswith("2") and name[:-1] in all_terms:
                pairs.add((name[:-1], name))

        normalized = []
        for a, b in pairs:
            if "left" in a and "right" in b:
                normalized.append((a, b))
            elif "left" in b and "right" in a:
                normalized.append((b, a))
            else:
                normalized.append((a, b))
        return normalized

    def _sample_swap_mask(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        count = int(env_ids.sum().item())
        if count == 0:
            return
        rand = torch.rand((count,), device=self.device)
        self._swap_mask[env_ids] = rand < self._swap_prob

    def _swap_actions_inplace(self, actions: torch.Tensor, swap_mask: torch.Tensor) -> torch.Tensor:
        if not torch.any(swap_mask):
            return actions
        if not self._action_term_slices:
            return actions
        for left, right in self._swap_action_term_pairs:
            if left not in self._action_term_slices or right not in self._action_term_slices:
                continue
            left_slice = self._action_term_slices[left]
            right_slice = self._action_term_slices[right]
            tmp = actions[swap_mask, left_slice].clone()
            actions[swap_mask, left_slice] = actions[swap_mask, right_slice]
            actions[swap_mask, right_slice] = tmp
        return actions

    def _swap_obs_inplace(self, obs_dict: dict, swap_mask: torch.Tensor):
        if not torch.any(swap_mask):
            return
        for group_name, group_obs in obs_dict.items():
            if group_name not in self._obs_group_concat:
                continue
            if self._obs_group_concat[group_name]:
                if group_name not in self._obs_term_slices:
                    continue
                term_slices = self._obs_term_slices[group_name]
                for left, right in self._swap_obs_term_pairs:
                    if left not in term_slices or right not in term_slices:
                        continue
                    left_slice = term_slices[left]
                    right_slice = term_slices[right]
                    tmp = group_obs[swap_mask, left_slice].clone()
                    group_obs[swap_mask, left_slice] = group_obs[swap_mask, right_slice]
                    group_obs[swap_mask, right_slice] = tmp
            else:
                if not isinstance(group_obs, dict):
                    continue
                for left, right in self._swap_obs_term_pairs:
                    if left not in group_obs or right not in group_obs:
                        continue
                    tmp = group_obs[left][swap_mask].clone()
                    group_obs[left][swap_mask] = group_obs[right][swap_mask]
                    group_obs[right][swap_mask] = tmp

    def _swap_reward_terms_inplace(self, swap_mask: torch.Tensor):
        if not torch.any(swap_mask):
            return
        if not hasattr(self.unwrapped, "reward_manager"):
            return
        reward_manager = self.unwrapped.reward_manager
        if self._reward_term_indices and hasattr(reward_manager, "_step_reward"):
            for left_idx, right_idx in self._reward_term_indices:
                tmp = reward_manager._step_reward[swap_mask, left_idx].clone()
                reward_manager._step_reward[swap_mask, left_idx] = reward_manager._step_reward[
                    swap_mask, right_idx
                ]
                reward_manager._step_reward[swap_mask, right_idx] = tmp
        if self._reward_term_pairs and hasattr(reward_manager, "_episode_sums"):
            for left_name, right_name in self._reward_term_pairs:
                if left_name not in reward_manager._episode_sums or right_name not in reward_manager._episode_sums:
                    continue
                tmp = reward_manager._episode_sums[left_name][swap_mask].clone()
                reward_manager._episode_sums[left_name][swap_mask] = reward_manager._episode_sums[right_name][
                    swap_mask
                ]
                reward_manager._episode_sums[right_name][swap_mask] = tmp

    def _modify_action_space(self):
        if self.clip_actions is None:
            return
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
