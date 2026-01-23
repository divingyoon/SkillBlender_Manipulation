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

import gymnasium as gym

from openarm.tasks.manager_based.openarm_manipulation.bimanual.grasp_2g_ik.config import agents as grasp2g_agents


gym.register(
    id="Grasp2g_IK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:OpenArmGrasp2gIKEnvCfg",
        "rl_games_cfg_entry_point": f"{grasp2g_agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{grasp2g_agents.__name__}.rsl_rl_ppo_cfg:OpenArmGrasp2gPPORunnerCfg",
        "rsl_rl_hier_cfg_entry_point": f"{grasp2g_agents.__name__}.rsl_rl_hierarchical_cfg:OpenArmGrasp2gHierarchicalPPORunnerCfg",
        "skrl_cfg_entry_point": f"{grasp2g_agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
