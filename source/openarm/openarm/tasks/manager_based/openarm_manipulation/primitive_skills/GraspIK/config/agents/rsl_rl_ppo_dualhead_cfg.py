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
        init_noise_std=0.5,
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
        # - 양손 비대칭 학습 문제 해결 + 붕괴 방지: 0.02 → 0.03
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=8,
        # [붕괴 방지] Learning rate 감소: 1e-4 → 5e-5
        # Phase 전환 시 급격한 policy 변화 완화
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        # [붕괴 방지] GAE lambda 감소: 0.95 → 0.9
        # Value function 오차에 대한 의존도 감소
        lam=0.95,
        # [붕괴 방지] KL 목표 완화: 0.01 → 0.02
        # Adaptive LR 감소 속도 완화 (LR 급감 방지)
        desired_kl=0.01,
        # [붕괴 방지] Gradient clipping 강화: 1.0 → 0.5
        # Phase 전환 시 gradient 폭발 방지
        max_grad_norm=0.5,
    )
