# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0

"""PPO training configuration for Primitive C: Tangential Compliance."""

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab.utils import configclass


__all__ = ["TangentialCompliancePPORunnerCfg"]


@configclass
class TangentialCompliancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for tangential compliance (directional slip control)."""

    num_steps_per_env = 32
    max_iterations = 2500
    save_interval = 250
    experiment_name = "primitive_c_tangential_compliance"
    run_name = ""
    resume = False
    empirical_normalization = True

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
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=8.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=1.0,
    )
