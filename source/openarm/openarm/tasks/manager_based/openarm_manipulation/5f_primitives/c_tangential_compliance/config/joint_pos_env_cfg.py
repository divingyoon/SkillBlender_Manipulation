# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0

"""OpenArm-specific configuration for Primitive C: Tangential Compliance."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .. import mdp
from ..tangential_compliance_env_cfg import TangentialComplianceEnvCfg

from ...common.robot_cfg import LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS
from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR


__all__ = ["OpenArmTangentialComplianceEnvCfg", "OpenArmTangentialComplianceEnvCfg_PLAY"]


@configclass
class OpenArmTangentialComplianceEnvCfg(TangentialComplianceEnvCfg):
    """OpenArm-specific configuration."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{OPENARM_ROOT_DIR}/usds/openarm_bimanual/openarm_tesollo_t3.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "openarm_left_joint1": 0.0,
                    "openarm_left_joint2": 0.0,
                    "openarm_left_joint3": 0.0,
                    "openarm_left_joint4": 0.0,
                    "openarm_left_joint5": 0.0,
                    "openarm_left_joint6": 0.0,
                    "openarm_left_joint7": 0.0,
                    "openarm_right_joint1": 0.0,
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

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.CylinderCfg(
                radius=0.025,
                height=0.08,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.15)),
        )

        self.scene.left_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/tesollo_left_.*_sensor_link",
            history_length=3,
            track_air_time=False,
        )

        self.scene.right_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/tesollo_right_.*_sensor_link",
            history_length=3,
            track_air_time=False,
        )

        self.actions.left_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=LEFT_HAND_JOINTS, scale=0.3, use_default_offset=True
        )

        self.actions.right_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=RIGHT_HAND_JOINTS, scale=0.3, use_default_offset=True
        )


@configclass
class OpenArmTangentialComplianceEnvCfg_PLAY(OpenArmTangentialComplianceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
