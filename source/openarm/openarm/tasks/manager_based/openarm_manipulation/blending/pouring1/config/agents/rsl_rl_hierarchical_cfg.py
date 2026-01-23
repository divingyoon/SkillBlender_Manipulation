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

import os

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass

from sbm.rl import SbmHierarchicalActorCriticCfg
from sbm.skill_registry import load_skill_registry

_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../../../../..")
)
os.environ.setdefault("SBM_SKILL_LOG_PATH", os.path.join(_ROOT_DIR, "obs_debug_pouring1.log"))


@configclass
class Pouring1HierarchicalPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    num_steps_per_env = 32
    max_iterations = 10000
    save_interval = 100
    experiment_name = "openarm_bi_pouring1_hier"
    run_name = ""
    resume = False
    empirical_normalization = True
    policy = SbmHierarchicalActorCriticCfg(
        init_noise_std=0.2,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        skill_dict=load_skill_registry(include=["openarm_bi_reach", "openarm_bi_grasp_2g"]),
        frame_stack=1,
        command_dim=14,
        command_slice=(0, 14),
        low_level_obs_groups=None,
        num_dofs=None,
        disable_skill_selection_until_iter=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )


@configclass
class Pouring1HierarchicalReachOnlyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Hierarchical runner using only the reach skill (debugging right-hand behavior)."""

    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    num_steps_per_env = 32
    max_iterations = 10000
    save_interval = 100
    experiment_name = "openarm_bi_pouring1_hier_reach_only"
    run_name = ""
    resume = False
    empirical_normalization = True
    policy = SbmHierarchicalActorCriticCfg(
        init_noise_std=0.2,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        skill_dict=load_skill_registry(include=["openarm_bi_reach"]),
        frame_stack=1,
        command_dim=14,
        command_slice=(0, 14),
        low_level_obs_groups=None,
        num_dofs=None,
        disable_skill_selection_until_iter=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )


@configclass
class Pouring1HierarchicalGraspOnlyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Hierarchical runner using only the grasp_2g skill (debugging right-hand behavior)."""

    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    num_steps_per_env = 32
    max_iterations = 10000
    save_interval = 100
    experiment_name = "openarm_bi_pouring1_hier_grasp_only"
    run_name = ""
    resume = False
    empirical_normalization = True
    policy = SbmHierarchicalActorCriticCfg(
        init_noise_std=0.2,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        skill_dict=load_skill_registry(include=["openarm_bi_grasp_2g"]),
        frame_stack=1,
        command_dim=14,
        command_slice=(0, 14),
        low_level_obs_groups=None,
        num_dofs=None,
        disable_skill_selection_until_iter=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )
