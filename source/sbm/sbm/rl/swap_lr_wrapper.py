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

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class SwapLRWrapper(gym.Wrapper):
    """Left/right swap wrapper for Isaac Lab envs (for skrl training)."""

    def __init__(
        self,
        env: ManagerBasedRLEnv | DirectRLEnv,
        swap_lr: bool = False,
        swap_prob: float = 0.5,
        swap_obs_term_pairs: list[tuple[str, str]] | None = None,
        swap_action_term_pairs: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(env)
        self._swap_lr = bool(swap_lr)
        self._swap_prob = float(swap_prob)

        # store info
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device

        self._swap_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._obs_term_slices: dict[str, dict[str, slice]] = {}
        self._obs_group_concat: dict[str, bool] = {}
        self._action_term_slices: dict[str, slice] = {}
        self._reward_term_pairs: list[tuple[str, str]] = []
        self._reward_term_indices: list[tuple[int, int]] = []

        self._swap_obs_term_pairs = swap_obs_term_pairs or []
        self._swap_action_term_pairs = swap_action_term_pairs or [
            ("left_arm_action", "right_arm_action"),
            ("left_hand_action", "right_hand_action"),
        ]

        if self._swap_lr:
            self._build_swap_helpers()
            if not self._swap_obs_term_pairs:
                self._swap_obs_term_pairs = self._infer_obs_swap_pairs()
            if self._swap_obs_term_pairs:
                pairs_str = ", ".join(f"{a}<->{b}" for a, b in self._swap_obs_term_pairs)
                print(f"[SWAP] obs pairs: {pairs_str}")
            self._sample_swap_mask()

    def reset(self, **kwargs):
        obs_dict, extras = self.env.reset(**kwargs)
        if self._swap_lr:
            self._sample_swap_mask()
            self._swap_obs_inplace(obs_dict, self._swap_mask)
        return obs_dict, extras

    def step(self, actions):
        if self._swap_lr:
            actions = self._swap_actions_inplace(actions, self._swap_mask)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        if self._swap_lr:
            self._swap_reward_terms_inplace(self._swap_mask)
            dones = (terminated | truncated)
            if torch.any(dones):
                self._sample_swap_mask(dones)
            self._swap_obs_inplace(obs_dict, self._swap_mask)
        return obs_dict, rew, terminated, truncated, extras

    def _build_swap_helpers(self):
        if hasattr(self.unwrapped, "observation_manager"):
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
        # left/right naming
        for name in list(all_terms):
            if "left" in name:
                cand = name.replace("left", "right")
                if cand in all_terms:
                    pairs.add((name, cand))
            if "right" in name:
                cand = name.replace("right", "left")
                if cand in all_terms:
                    pairs.add((cand, name))
        # handle *_2 suffix patterns
        for name in list(all_terms):
            if name.endswith("2") and name[:-1] in all_terms:
                pairs.add((name[:-1], name))
        # handle cup/cup2 with left/right split in the middle (grasp2g naming)
        for name in list(all_terms):
            if "cup2" in name:
                cand = name.replace("cup2", "cup")
                cand = cand.replace("right", "left")
                if cand in all_terms:
                    pairs.add((cand, name))
            if "cup" in name and "cup2" not in name:
                cand = name.replace("cup", "cup2")
                cand = cand.replace("left", "right")
                if cand in all_terms:
                    pairs.add((name, cand))
        # handle object/object2 similarly
        for name in list(all_terms):
            if "object2" in name:
                cand = name.replace("object2", "object")
                cand = cand.replace("right", "left")
                if cand in all_terms:
                    pairs.add((cand, name))
            if "object" in name and "object2" not in name:
                cand = name.replace("object", "object2")
                cand = cand.replace("left", "right")
                if cand in all_terms:
                    pairs.add((name, cand))

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
