# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0

"""OpenArm-specific configuration for Primitive D: Finger Synergy Shape."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .. import mdp
from ..finger_synergy_shape_env_cfg import FingerSynergyShapeEnvCfg

from ...common.robot_cfg import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS
from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR


__all__ = ["OpenArmFingerSynergyShapeEnvCfg", "OpenArmFingerSynergyShapeEnvCfg_PLAY"]


@configclass
class OpenArmFingerSynergyShapeEnvCfg(FingerSynergyShapeEnvCfg):
    """OpenArm-specific configuration."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"/home/user/rl_ws/robot_setting/openarm_tesollo_mount/openarm_tesollo_mount.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    fix_root_link=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 1.0),
                joint_pos={
                    "openarm_left_joint1": -1.3,
                    "openarm_left_joint2": 0.0,
                    "openarm_left_joint3": 0.0,
                    "openarm_left_joint4": 0.0,
                    "openarm_left_joint5": 0.0,
                    "openarm_left_joint6": 0.0,
                    "openarm_left_joint7": 0.0,
                    "openarm_right_joint1": 1.3,
                    "openarm_right_joint2": 0.0,
                    "openarm_right_joint3": 0.0,
                    "openarm_right_joint4": 0.0,
                    "openarm_right_joint5": 0.0,
                    "openarm_right_joint6": 0.0,
                    "openarm_right_joint7": 0.0,
                    "lj_dg_.*": 0.0,
                    "rj_dg_.*": 0.0,
                },
            ),
            actuators={
                "openarm_arm": ImplicitActuatorCfg(
                    joint_names_expr=[
                        "openarm_left_joint[1-7]",
                        "openarm_right_joint[1-7]",
                    ],
                    stiffness=400.0,
                    damping=80.0,
                ),
                "openarm_gripper": ImplicitActuatorCfg(
                    joint_names_expr=[
                        "lj_dg_.*",
                        "rj_dg_.*",
                    ],
                    stiffness=2e3,
                    damping=1e2,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )

        self.actions.left_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEFT_HAND_JOINTS,
            scale=0.4,
            use_default_offset=True,
        )

        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEFT_ARM_JOINTS,
            scale=0.0,
            use_default_offset=True,
        )
        self.actions.left_arm_action.use_default_offset = False
        self.actions.left_arm_action.offset = {
            "openarm_left_joint1": -1.3,
        }

        self.actions.right_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_HAND_JOINTS,
            scale=0.4,
            use_default_offset=True,
        )

        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_ARM_JOINTS,
            scale=0.0,
            use_default_offset=True,
        )
        self.actions.right_arm_action.use_default_offset = False
        self.actions.right_arm_action.offset = {
            "openarm_right_joint1": 1.3,
        }



@configclass
class OpenArmFingerSynergyShapeEnvCfg_PLAY(OpenArmFingerSynergyShapeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
