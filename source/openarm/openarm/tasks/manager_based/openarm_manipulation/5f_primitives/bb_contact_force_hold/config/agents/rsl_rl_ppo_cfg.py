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

"""PPO training configuration for Primitive B: Contact Force Hold."""

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from isaaclab.utils import configclass


__all__ = ["ContactForceHoldPPORunnerCfg"]


@configclass
class ContactForceHoldPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for contact force hold primitive.

    This is a more complex contact-based task, so we use:
    - Larger networks (256, 128, 64)
    - More training iterations
    - Higher entropy for exploration
    """

    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 50
    experiment_name = "primitive_bb_contact_force_hold"
    run_name = ""
    resume = False
    empirical_normalization = True  # Important for force observations

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Higher entropy for exploration
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-4,  # Lower LR for stability
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
