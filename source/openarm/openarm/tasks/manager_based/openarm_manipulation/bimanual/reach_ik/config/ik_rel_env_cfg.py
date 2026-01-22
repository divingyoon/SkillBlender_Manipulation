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

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from openarm.tasks.manager_based.openarm_manipulation.bimanual.reach.config import joint_pos_env_cfg as reach_joint_cfg
from openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import OPEN_ARM_HIGH_PD_CFG


@configclass
class OpenArmReachIKEnvCfg(reach_joint_cfg.OpenArmReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Keep the high-PD robot for stable IK tracking, preserve init_state from parent
        self.scene.robot = OPEN_ARM_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=self.scene.robot.init_state,
        )

        # Left arm IK (relative)
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_joint[1-7]"],
            body_name="openarm_left_ee_tcp",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
        )

        # Right arm IK (relative)
        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_joint[1-7]"],
            body_name="openarm_right_ee_tcp",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
        )
