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

from openarm.tasks.manager_based.openarm_manipulation.bimanual.grasp_2g_ik.config import (
    joint_pos_env_cfg as grasp2g_joint_cfg,
)
from .. import mdp


@configclass
class OpenArmGrasp2gIKEnvCfg(grasp2g_joint_cfg.OpenArmGrasp2gEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # DualHead를 위해 Left → Right 순서로 배치
        # [0:6] left_arm (IK pose), [6:8] left_hand, [8:14] right_arm (IK pose), [14:16] right_hand
        # dof_split_index = 8 (left: 0~7, right: 8~15)

        # Left arm IK (relative)
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_joint[1-7]"],
            body_name="openarm_left_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.2,
        )
        # Left hand (gripper)
        self.actions.left_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_finger_joint.*"],
            scale=0.15,
            use_default_offset=True,
        )

        # Right arm IK (relative)
        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_joint[1-7]"],
            body_name="openarm_right_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.2,
        )
        # Right hand (gripper)
        self.actions.right_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],
            scale=0.15,
            use_default_offset=True,
        )
