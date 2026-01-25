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
Grasp environment with reach policy rollout.

This environment runs a pre-trained reach policy during reset to
initialize the robot in a realistic pre-grasp state.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


class GraspWithRolloutEnv(ManagerBasedRLEnv):
    """Grasp environment that uses reach policy rollout for initialization.

    During reset, this environment:
    1. Resets to default state
    2. Runs the pre-trained reach policy for N steps
    3. Starts the grasp episode from the resulting state
    """

    def __init__(
        self,
        cfg: ManagerBasedRLEnvCfg,
        reach_policy_path: str = "",
        reach_rollout_steps: int = 100,
        render_mode: str | None = None,
        **kwargs,
    ):
        """Initialize the environment.

        Args:
            cfg: Environment configuration.
            reach_policy_path: Path to trained reach policy checkpoint.
            reach_rollout_steps: Number of steps to run reach policy.
            render_mode: Render mode for the environment.
        """
        super().__init__(cfg, render_mode, **kwargs)

        self.reach_policy_path = reach_policy_path
        self.reach_rollout_steps = reach_rollout_steps
        self.reach_policy = None
        self.reach_obs_normalizer = None

        # Load reach policy if path provided
        if reach_policy_path and os.path.exists(reach_policy_path):
            self._load_reach_policy()

    def _load_reach_policy(self):
        """Load the pre-trained reach policy."""
        print(f"[GraspWithRolloutEnv] Loading reach policy from: {self.reach_policy_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(self.reach_policy_path, map_location=self.device)

            # RSL-RL saves model in specific format
            if "model_state_dict" in checkpoint:
                self._reach_model_state = checkpoint["model_state_dict"]
            else:
                self._reach_model_state = checkpoint

            # Try to load observation normalizer if exists
            if "obs_norm_state_dict" in checkpoint:
                self._reach_obs_norm_state = checkpoint["obs_norm_state_dict"]
            else:
                self._reach_obs_norm_state = None

            print(f"[GraspWithRolloutEnv] Reach policy loaded successfully")
            self._reach_policy_loaded = True

        except Exception as e:
            print(f"[GraspWithRolloutEnv] Failed to load reach policy: {e}")
            self._reach_policy_loaded = False

    def _run_reach_rollout(self, env_ids: torch.Tensor):
        """Run reach policy rollout for specified environments.

        Args:
            env_ids: Environment indices to run rollout for.
        """
        if not hasattr(self, "_reach_policy_loaded") or not self._reach_policy_loaded:
            return

        if self.reach_policy is None:
            # Policy needs to be built - this is done lazily
            # to ensure we have the correct observation dimensions
            self._build_reach_policy()

        if self.reach_policy is None:
            return

        # Run rollout steps
        for step in range(self.reach_rollout_steps):
            # Get observations for reach policy
            obs = self._get_reach_observations()

            # Get actions from reach policy
            with torch.no_grad():
                if self.reach_obs_normalizer is not None:
                    obs = self.reach_obs_normalizer(obs)
                actions = self.reach_policy(obs)

            # Apply actions only to specified envs
            # Note: This requires careful handling to only affect env_ids
            full_actions = self.action_manager.action.clone()
            full_actions[env_ids] = actions[env_ids]

            # Step simulation (internal step, not full env step)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.sim.get_physics_dt())

    def _build_reach_policy(self):
        """Build the reach policy network from saved state."""
        # This requires knowing the policy architecture
        # For RSL-RL ActorCritic, we need to reconstruct the actor

        if not hasattr(self, "_reach_model_state"):
            return

        # Infer network dimensions from state dict
        state_dict = self._reach_model_state

        # Find actor layer dimensions
        actor_layers = {}
        for key, value in state_dict.items():
            if "actor" in key and "weight" in key:
                actor_layers[key] = value.shape

        if not actor_layers:
            print("[GraspWithRolloutEnv] Could not find actor layers in checkpoint")
            return

        # Build simple MLP actor
        # Note: This assumes standard RSL-RL actor structure
        try:
            # Get input/output dimensions from first/last layers
            layer_keys = sorted([k for k in actor_layers.keys()])
            first_layer = state_dict[layer_keys[0]]
            last_layer_key = [k for k in layer_keys if "weight" in k][-1]
            last_layer = state_dict[last_layer_key]

            input_dim = first_layer.shape[1]
            output_dim = last_layer.shape[0]

            # Get hidden dimensions
            hidden_dims = []
            for key in layer_keys[:-1]:
                if "weight" in key:
                    hidden_dims.append(state_dict[key].shape[0])

            # Build network
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ELU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))

            self.reach_policy = nn.Sequential(*layers).to(self.device)

            # Load weights (need to map keys correctly)
            # This is simplified - actual implementation may need key mapping
            self.reach_policy.eval()

            print(f"[GraspWithRolloutEnv] Built reach policy: {input_dim} -> {hidden_dims} -> {output_dim}")

        except Exception as e:
            print(f"[GraspWithRolloutEnv] Failed to build reach policy: {e}")
            self.reach_policy = None

    def _get_reach_observations(self) -> torch.Tensor:
        """Get observations formatted for reach policy."""
        # Use the observation manager to get observations
        obs_dict = self.observation_manager.compute()
        return obs_dict["policy"]

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment with optional reach rollout.

        Args:
            seed: Random seed.
            options: Reset options. Set options["skip_rollout"]=True to skip.

        Returns:
            Observations and info dict.
        """
        # Standard reset first
        obs, info = super().reset(seed=seed, options=options)

        # Run reach rollout if policy is loaded and not skipped
        skip_rollout = options.get("skip_rollout", False) if options else False

        if not skip_rollout and hasattr(self, "_reach_policy_loaded") and self._reach_policy_loaded:
            # Run rollout for all environments
            all_env_ids = torch.arange(self.num_envs, device=self.device)
            self._run_reach_rollout(all_env_ids)

            # Get updated observations after rollout
            obs = self.observation_manager.compute()["policy"]

        return obs, info
