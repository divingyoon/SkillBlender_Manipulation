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
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from openarm.tasks.manager_based.openarm_manipulation.primitive_skills.GraspIK.config import (
    joint_pos_env_cfg as grasp2g_joint_cfg,
)
from openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import OPEN_ARM_HIGH_PD_CFG
from .. import mdp


@configclass
class GraspIKIKEnvCfg(grasp2g_joint_cfg.GraspIKJointPosEnvCfg):
    # Toggle relative vs absolute IK commands
    use_relative_mode: bool = True

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Quick toggles for diagnosis
        enable_orientation_constraint = False
        enable_nullspace = True
        enable_joint_limit_avoidance = True

        # Keep the high-PD robot for stable IK tracking, use joint_pos_env_cfg init_state
        self.scene.robot = OPEN_ARM_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=[0.0, 0.0, -0.25],
                rot=[1.0, 0.0, 0.0, 0.0],
                joint_pos={
                    # Pre-grasp pose for left arm (similar to reach end pose)
                    "openarm_left_joint1": -0.5,
                    "openarm_left_joint2": -0.5,
                    "openarm_left_joint3": 0.6,
                    "openarm_left_joint4": 0.7,
                    "openarm_left_joint5": 0.0,
                    "openarm_left_joint6": 0.0,
                    "openarm_left_joint7": -1.0,
                    # Pre-grasp pose for right arm (mirrored)
                    "openarm_right_joint1": 0.5,
                    "openarm_right_joint2": 0.5,
                    "openarm_right_joint3": -0.6,
                    "openarm_right_joint4": 0.7,
                    "openarm_right_joint5": 0.0,
                    "openarm_right_joint6": 0.0,
                    "openarm_right_joint7": 1.0,
                    # Grippers fully open
                    "openarm_left_finger_joint1": 0.044,
                    "openarm_left_finger_joint2": 0.052,
                    "openarm_right_finger_joint1": 0.044,
                    "openarm_right_finger_joint2": 0.052,
                },
            ),
        )

        # DualHead를 위해 Left → Right 순서로 배치
        # [0:6] left_arm (IK pose), [6:8] left_hand, [8:14] right_arm (IK pose), [14:16] right_hand
        # dof_split_index = 8 (left: 0~7, right: 8~15)

        # Left arm IK (relative)
        self.actions.left_arm_action = mdp.ConstrainedDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_joint[1-7]"],
            body_name="openarm_left_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=self.use_relative_mode,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
            ),
            scale=0.05,
            orientation_constraint=enable_orientation_constraint,
            orientation_command_name="left_object_pose",
            orientation_object_axis=(0.0, 1.0, 0.0),
            orientation_roll=0.0,
            nullspace_gain=0.1 if enable_nullspace else 0.0,
            joint_limit_avoidance_gain=0.1 if enable_joint_limit_avoidance else 0.0,
            joint_limit_eps=1.0e-3,
            joint_limit_clamp=50.0,
        )
        # Left hand (gripper)
        self.actions.left_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_finger_joint.*"],
            scale=0.15,
            use_default_offset=True,
        )

        # Right arm IK (relative)
        self.actions.right_arm_action = mdp.ConstrainedDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_joint[1-7]"],
            body_name="openarm_right_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=self.use_relative_mode,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
            ),
            scale=0.05,
            orientation_constraint=enable_orientation_constraint,
            orientation_command_name="right_object_pose",
            orientation_object_axis=(0.0, 1.0, 0.0),
            orientation_roll=0.0,
            nullspace_gain=0.1 if enable_nullspace else 0.0,
            joint_limit_avoidance_gain=0.1 if enable_joint_limit_avoidance else 0.0,
            joint_limit_eps=1.0e-3,
            joint_limit_clamp=50.0,
        )
        # Right hand (gripper)
        self.actions.right_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],
            scale=0.15,
            use_default_offset=True,
        )
