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

"""OpenArm-specific configuration for Primitive B: Contact Force Hold."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .. import mdp
from ..contact_force_hold_env_cfg import ContactForceHoldEnvCfg

from ...common.robot_cfg import (
    LEFT_ARM_JOINTS,
    LEFT_CONTACT_LINKS,
    LEFT_HAND_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_CONTACT_LINKS,
    RIGHT_HAND_JOINTS,
)
from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR


__all__ = [
    "OpenArmContactForceHoldEnvCfg",
    "OpenArmContactForceHoldEnvCfg_PLAY",
]


@configclass
class OpenArmContactForceHoldEnvCfg(ContactForceHoldEnvCfg):
    """OpenArm-specific configuration for contact force hold primitive."""

    def __post_init__(self):
        # Call parent post_init
        super().__post_init__()

        # Set robot
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/openarm_tesollo_mount",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"/home/user/rl_ws/robot_setting/openarm_tesollo_mount/openarm_tesollo_mount.usd",
                activate_contact_sensors=True,
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
        self.scene.robot.init_state.pos = (0.10, 0.0, -0.20)

        # Set graspable object (left, cylinder/box/cup proxy)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[
                    # Cylinder (base radius; scaled continuously at prestartup).
                    sim_utils.CylinderCfg(
                        radius=0.05,
                        height=0.16,
                        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
                    ),
                    # Box (base size; scaled continuously at prestartup).
                    sim_utils.CuboidCfg(
                        size=(0.08, 0.08, 0.14),
                        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
                    ),
                    # Cone
                    # sim_utils.ConeCfg(
                    #     radius=0.055,
                    #     height=0.16,
                    #     physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
                    # ),
                ],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.43, 0.08, 0.14),  # 5cm more forward from left grasp center
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # Set graspable object (right, cylinder/box/cup proxy)
        self.scene.object_right = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ObjectRight",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[
                    sim_utils.CylinderCfg(
                        radius=0.05,
                        height=0.16,
                        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
                    ),
                    sim_utils.CuboidCfg(
                        size=(0.08, 0.08, 0.14),
                        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
                    ),
                ],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.43, -0.08, 0.14),  # mirrored from left grasp center
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # Per-fingertip contact sensors filtered to the object.
        for idx, link_name in enumerate(LEFT_CONTACT_LINKS, start=1):
            sensor_name = "left_contact_sensor" if idx == 1 else f"left_contact_sensor_{idx}"
            setattr(
                self.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/openarm_tesollo_mount/tesollo_left/{link_name}",
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object.*"],
                    history_length=3,
                    track_air_time=False,
                ),
            )

        for idx, link_name in enumerate(RIGHT_CONTACT_LINKS, start=1):
            sensor_name = "right_contact_sensor" if idx == 1 else f"right_contact_sensor_{idx}"
            setattr(
                self.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/openarm_tesollo_mount/tesollo_right/{link_name}",
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/ObjectRight.*"],
                    history_length=3,
                    track_air_time=False,
                ),
            )


        # Debug: visualize left fingertip frames (axes) in the viewport.
        left_tip_source = LEFT_CONTACT_LINKS[0]
        self.scene.left_tip_frames = FrameTransformerCfg(
            prim_path=f"{{ENV_REGEX_NS}}/openarm_tesollo_mount/tesollo_left/{left_tip_source}",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/openarm_tesollo_mount/tesollo_left/{link_name}",
                    name=f"left_tip_{link_name}",
                )
                for link_name in LEFT_CONTACT_LINKS
            ],
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/Debug/LeftTipFrames", markers={"frame": sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd", scale=(0.025, 0.025, 0.025)), "connecting_line": sim_utils.CylinderCfg(radius=0.0001, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), roughness=1.0))}),
        )


        # Set actions
        self.actions.left_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEFT_HAND_JOINTS,
            scale=0.2,
            use_default_offset=False,
            offset={
                **{name: 0.0 for name in LEFT_HAND_JOINTS},
                "lj_dg_1_2": 1.22173,
                "lj_dg_2_2": 0.10,
                "lj_dg_3_2": 0.10,
                "lj_dg_4_2": 0.10,
                "lj_dg_5_2": 0.10,
                "lj_dg_1_3": -0.25,
                "lj_dg_2_3": 0.7164,
                "lj_dg_3_3": 0.7164,
                "lj_dg_4_3": 0.7164,
                "lj_dg_5_3": 0.5419,
                "lj_dg_1_4": -0.15,
                "lj_dg_2_4": 0.3245,
                "lj_dg_3_4": 0.3245,
                "lj_dg_4_4": 0.3245,
                "lj_dg_5_4": 0.1500,
            },
        )

        self.actions.right_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_HAND_JOINTS,
            scale=0.2,
            use_default_offset=False,
            offset={
                **{name: 0.0 for name in RIGHT_HAND_JOINTS},
                "rj_dg_1_2": -1.22173,
                "rj_dg_2_2": 0.10,
                "rj_dg_3_2": 0.10,
                "rj_dg_4_2": 0.10,
                "rj_dg_5_2": 0.10,
                "rj_dg_1_3": 0.25,
                "rj_dg_2_3": 0.7164,
                "rj_dg_3_3": 0.7164,
                "rj_dg_4_3": 0.7164,
                "rj_dg_5_3": 0.5419,
                "rj_dg_1_4": 0.15,
                "rj_dg_2_4": 0.3245,
                "rj_dg_3_4": 0.3245,
                "rj_dg_4_4": 0.3245,
                "rj_dg_5_4": 0.1500,
            },
        )

        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[name for name in LEFT_ARM_JOINTS if name not in {"openarm_left_joint1", "openarm_left_joint2"}],
            scale={
                "openarm_left_joint3": 0.0,
                "openarm_left_joint4": 0.0,
                "openarm_left_joint5": 0.2,
                "openarm_left_joint6": 0.2,
                "openarm_left_joint7": 0.2,
            },
            use_default_offset=False,
            offset={
                "openarm_left_joint3": -0.2,
                "openarm_left_joint4": 0.7745,
                "openarm_left_joint5": 0.0,
                "openarm_left_joint6": 0.1,
                "openarm_left_joint7": -0.7,
            },
        )

        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_ARM_JOINTS,
            scale={
                "openarm_right_joint3": 0.0,
                "openarm_right_joint4": 0.0,
                "openarm_right_joint5": 0.2,
                "openarm_right_joint6": 0.2,
                "openarm_right_joint7": 0.2,
            },
            use_default_offset=False,
            offset={
                "openarm_right_joint3": 0.2,
                "openarm_right_joint4": 0.7745,
                "openarm_right_joint5": 0.0,
                "openarm_right_joint6": -0.1,
                "openarm_right_joint7": 0.7,
            },
        )


@configclass
class OpenArmContactForceHoldEnvCfg_PLAY(OpenArmContactForceHoldEnvCfg):
    """Play/evaluation configuration."""

    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for visualization
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0

        # Disable observation noise
        self.observations.policy.enable_corruption = False
