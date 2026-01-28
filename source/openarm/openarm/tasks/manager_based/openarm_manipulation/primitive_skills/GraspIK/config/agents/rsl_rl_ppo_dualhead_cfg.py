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

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass

from sbm.rl import SbmDualHeadActorCriticCfg


@configclass
class Grasp2gIKDualHeadPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "GraspIK-v0"
    run_name = ""
    resume = False
    empirical_normalization = False
    policy = SbmDualHeadActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256],
        critic_hidden_dims=[512, 256],
        activation="elu",
        # IK 환경: left_arm(6) + left_hand(2) = 8, right_arm(6) + right_hand(2) = 8
        # Total: 16, split at index 8
        dof_split_index=8,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # [방법2] 탐험(Exploration) 파라미터
        # entropy_coef: 정책의 엔트로피에 대한 가중치
        # - 높을수록 더 많은 탐험 (다양한 행동 시도)
        # - 낮을수록 exploitation 위주 (검증된 행동 반복)
        # - 양손 비대칭 학습 문제 해결: 0.01 → 0.02로 증가하여
        #   한쪽이 성공해도 다른쪽이 독립적으로 탐험하도록 유도
        entropy_coef=0.02,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
