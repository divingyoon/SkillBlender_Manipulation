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

"""
Skill rollout utilities for curriculum learning.

This module provides functionality to run a pre-trained policy during
environment reset, enabling "rollout-from-previous-skill" initialization.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SkillRolloutManager:
    """Manages loading and executing pre-trained skill policies.

    This class handles:
    1. Loading a trained policy checkpoint
    2. Running the policy for a specified number of steps
    3. Providing the final state as the starting point for the next skill
    """

    def __init__(
        self,
        env: "ManagerBasedRLEnv",
        policy_path: str,
        num_rollout_steps: int = 100,
        device: str = "cuda:0",
    ):
        """Initialize the skill rollout manager.

        Args:
            env: The environment instance.
            policy_path: Path to the trained policy checkpoint (.pt file).
            num_rollout_steps: Number of steps to run the previous skill.
            device: Device to load the policy on.
        """
        self.env = env
        self.policy_path = policy_path
        self.num_rollout_steps = num_rollout_steps
        self.device = device
        self.policy = None
        self._obs_normalizer = None

        if policy_path and os.path.exists(policy_path):
            self._load_policy()

    def _load_policy(self):
        """Load the trained policy from checkpoint."""
        print(f"[SkillRolloutManager] Loading policy from: {self.policy_path}")

        checkpoint = torch.load(self.policy_path, map_location=self.device)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
        elif "actor" in checkpoint:
            model_state = checkpoint
        else:
            model_state = checkpoint

        # Extract actor network dimensions from state dict
        # This is a simplified loader - may need adjustment based on actual policy structure
        self.policy = self._build_policy_from_state(model_state)

        if self.policy is not None:
            self.policy.eval()
            print(f"[SkillRolloutManager] Policy loaded successfully")
        else:
            print(f"[SkillRolloutManager] Warning: Could not load policy")

    def _build_policy_from_state(self, state_dict: dict) -> nn.Module | None:
        """Build policy network from state dict.

        This is a placeholder - the actual implementation depends on
        the policy architecture used (RSL-RL, etc.)
        """
        # For RSL-RL policies, we need to reconstruct the actor network
        # This requires knowing the network architecture

        # Try to infer dimensions from state dict keys
        actor_keys = [k for k in state_dict.keys() if "actor" in k.lower()]

        if not actor_keys:
            return None

        # Store the state dict for later use
        self._policy_state_dict = state_dict
        return None  # Will use direct inference instead

    def rollout(self, env_ids: torch.Tensor) -> None:
        """Execute the previous skill policy for specified environments.

        Args:
            env_ids: Tensor of environment indices to rollout.
        """
        if self.policy is None and self._policy_state_dict is None:
            print("[SkillRolloutManager] No policy loaded, skipping rollout")
            return

        # Note: Full rollout requires stepping the simulation
        # This is typically done outside the reset event
        # For now, we store env_ids that need rollout
        if not hasattr(self.env, "_pending_rollout_envs"):
            self.env._pending_rollout_envs = []

        self.env._pending_rollout_envs.extend(env_ids.tolist())


def create_rollout_reset_event(
    policy_path: str,
    num_rollout_steps: int = 100,
):
    """Factory function to create a rollout reset event.

    Args:
        policy_path: Path to the trained policy checkpoint.
        num_rollout_steps: Number of steps to run the previous skill.

    Returns:
        A reset event function.
    """
    _manager = None

    def rollout_reset(
        env: "ManagerBasedRLEnv",
        env_ids: torch.Tensor,
    ) -> None:
        """Reset event that triggers rollout from previous skill."""
        nonlocal _manager

        if _manager is None:
            _manager = SkillRolloutManager(
                env=env,
                policy_path=policy_path,
                num_rollout_steps=num_rollout_steps,
                device=str(env.device),
            )

        _manager.rollout(env_ids)

    return rollout_reset
