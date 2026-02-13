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

import torch

from isaaclab.envs import ManagerBasedRLEnv


class Lift5gHoldEnv(ManagerBasedRLEnv):
    """Lift env with 2g-style inactive-arm action masking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_inactive_arm = getattr(self.cfg, "mask_inactive_arm_actions", True)
        self._reward_stage = -1
        self._setup_curriculum_masking()

    def _setup_curriculum_masking(self) -> None:
        self._action_slices: dict[str, slice] = {}
        idx = 0
        for name, term in self.action_manager._terms.items():
            dim = term.action_dim
            self._action_slices[name] = slice(idx, idx + dim)
            idx += dim

        if "left_arm_action" in self._action_slices:
            sl = self._action_slices["left_arm_action"]
            self._left_arm_hold = torch.zeros(self.num_envs, sl.stop - sl.start, device=self.device)
        if "right_arm_action" in self._action_slices:
            sl = self._action_slices["right_arm_action"]
            self._right_arm_hold = torch.zeros(self.num_envs, sl.stop - sl.start, device=self.device)
        if "left_hand_action" in self._action_slices:
            sl = self._action_slices["left_hand_action"]
            self._left_hand_hold = torch.zeros(self.num_envs, sl.stop - sl.start, device=self.device)
        if "right_hand_action" in self._action_slices:
            sl = self._action_slices["right_hand_action"]
            self._right_hand_hold = torch.zeros(self.num_envs, sl.stop - sl.start, device=self.device)
        if "left_thumb_action" in self._action_slices:
            sl = self._action_slices["left_thumb_action"]
            self._left_thumb_hold = torch.zeros(self.num_envs, sl.stop - sl.start, device=self.device)
        if "right_thumb_action" in self._action_slices:
            sl = self._action_slices["right_thumb_action"]
            self._right_thumb_hold = torch.zeros(self.num_envs, sl.stop - sl.start, device=self.device)

    def _get_curriculum_stage(self) -> int:
        return int(getattr(self.cfg, "curriculum_stage", 2))

    def _set_reward_weight(self, term_name: str, weight: float) -> None:
        if not hasattr(self, "reward_manager"):
            return
        if not hasattr(self.reward_manager, "get_term_cfg") or not hasattr(self.reward_manager, "set_term_cfg"):
            return
        try:
            term_cfg = self.reward_manager.get_term_cfg(term_name)
        except Exception:
            return
        if hasattr(term_cfg, "weight") and abs(float(term_cfg.weight) - float(weight)) > 1e-9:
            term_cfg.weight = float(weight)
            self.reward_manager.set_term_cfg(term_name, term_cfg)

    def _apply_staged_reward_curriculum(self) -> None:
        step = int(getattr(self, "common_step_counter", 0))
        stage1 = int(getattr(self.cfg, "reward_stage_1_step", 20_000))
        stage2 = int(getattr(self.cfg, "reward_stage_2_step", 50_000))
        stage3 = int(getattr(self.cfg, "reward_stage_3_step", 90_000))

        if step < stage1:
            stage = 0
        elif step < stage2:
            stage = 1
        elif step < stage3:
            stage = 2
        else:
            stage = 3

        if stage == self._reward_stage:
            return
        self._reward_stage = stage

        # Stage-wise weights: approach -> grasp -> lift -> goal tracking.
        if stage == 0:
            weights = {
                "finger_grasp": 0.0,
                "contact_persistence": 0.0,
                "slip_penalty": 0.0,
                "normal_force_stability": 0.0,
                "force_spike": 0.0,
                "overgrip": 0.0,
                "lifting_object": 0.0,
                "object_goal_tracking": 0.0,
                "object_goal_tracking_fine_grained": 0.0,
            }
        elif stage == 1:
            weights = {
                "finger_grasp": 10.0,
                "contact_persistence": 6.0,
                "slip_penalty": -0.4,
                "normal_force_stability": 0.2,
                "force_spike": -0.05,
                "overgrip": -0.2,
                "lifting_object": 12.0,
                "object_goal_tracking": 0.0,
                "object_goal_tracking_fine_grained": 0.0,
            }
        elif stage == 2:
            weights = {
                "finger_grasp": 12.0,
                "contact_persistence": 7.0,
                "slip_penalty": -0.5,
                "normal_force_stability": 0.6,
                "force_spike": -0.05,
                "overgrip": -0.2,
                "lifting_object": 16.0,
                "object_goal_tracking": 10.0,
                "object_goal_tracking_fine_grained": 5.0,
            }
        else:
            weights = {
                "finger_grasp": 14.0,
                "contact_persistence": 8.0,
                "slip_penalty": -0.6,
                "normal_force_stability": 1.0,
                "force_spike": -0.1,
                "overgrip": -0.3,
                "lifting_object": 20.0,
                "object_goal_tracking": 20.0,
                "object_goal_tracking_fine_grained": 10.0,
            }

        for term_name, weight in weights.items():
            self._set_reward_weight(term_name, weight)

    def step(self, action: torch.Tensor):
        action = action.to(self.device)

        if self.mask_inactive_arm:
            stage = self._get_curriculum_stage()

            if stage == 0:
                if "right_arm_action" in self._action_slices:
                    action[:, self._action_slices["right_arm_action"]] = self._right_arm_hold
                if "right_hand_action" in self._action_slices:
                    action[:, self._action_slices["right_hand_action"]] = self._right_hand_hold
                if "right_thumb_action" in self._action_slices:
                    action[:, self._action_slices["right_thumb_action"]] = self._right_thumb_hold

            elif stage == 1:
                if "left_arm_action" in self._action_slices:
                    action[:, self._action_slices["left_arm_action"]] = self._left_arm_hold
                if "left_hand_action" in self._action_slices:
                    action[:, self._action_slices["left_hand_action"]] = self._left_hand_hold
                if "left_thumb_action" in self._action_slices:
                    action[:, self._action_slices["left_thumb_action"]] = self._left_thumb_hold

        return super().step(action)
