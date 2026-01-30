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


class Grasp2gHoldEnv(ManagerBasedRLEnv):
    """Grasp env that holds gripper closed once grasp phase begins."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_gripper_hold = getattr(self.cfg, "enable_gripper_hold", False)
        self._setup_gripper_hold()

    def _setup_gripper_hold(self) -> None:
        self._action_slices: dict[str, slice] = {}
        idx = 0
        for name, term in self.action_manager._terms.items():
            dim = term.action_dim
            self._action_slices[name] = slice(idx, idx + dim)
            idx += dim

        self._left_hand_term = self.action_manager.get_term("left_hand_action")
        self._right_hand_term = self.action_manager.get_term("right_hand_action")

        self._left_close_raw = self._compute_close_raw(self._left_hand_term)
        self._right_close_raw = self._compute_close_raw(self._right_hand_term)

    def _compute_close_raw(self, term) -> torch.Tensor:
        joint_ids = term._joint_ids
        joint_limits = term._asset.data.joint_pos_limits
        target = joint_limits[:, joint_ids, 0]

        if isinstance(term._offset, torch.Tensor):
            offset = term._offset
        else:
            offset = torch.full_like(target, float(term._offset))

        if isinstance(term._scale, torch.Tensor):
            scale = term._scale
        else:
            scale = torch.full_like(target, float(term._scale))

        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        return (target - offset) / scale

    def _get_phase(self, attr_name: str) -> torch.Tensor:
        if hasattr(self, attr_name):
            phase = getattr(self, attr_name)
            if isinstance(phase, torch.Tensor):
                return phase
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def step(self, action: torch.Tensor):
        if not self.enable_gripper_hold:
            return super().step(action)

        action = action.to(self.device)

        left_phase = self._get_phase("grasp2g_phase_left")
        right_phase = self._get_phase("grasp2g_phase_right")

        if "left_hand_action" in self._action_slices:
            left_mask = left_phase >= 1
            if torch.any(left_mask):
                sl = self._action_slices["left_hand_action"]
                action[left_mask, sl] = self._left_close_raw[left_mask]

        if "right_hand_action" in self._action_slices:
            right_mask = right_phase >= 1
            if torch.any(right_mask):
                sl = self._action_slices["right_hand_action"]
                action[right_mask, sl] = self._right_close_raw[right_mask]

        return super().step(action)

    def set_gripper_hold(self, enabled: bool) -> None:
        """Toggle gripper hold behavior at runtime."""
        self.enable_gripper_hold = bool(enabled)
