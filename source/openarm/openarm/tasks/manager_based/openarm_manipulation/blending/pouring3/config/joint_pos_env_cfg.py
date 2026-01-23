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

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR
from openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import (
    OPEN_ARM_HIGH_PD_CFG,
)

from .. import mdp
from ..pouring_env_cfg import Pouring3BaseEnvCfg


@configclass
class Pouring3EnvCfg(Pouring3BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Disable one-time IK reset; use fixed joint init from pouring2 baseline.
        self.events.reset_robot_tcp_to_cups = None

        self.scene.robot = OPEN_ARM_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=[0.0, 0.0, -0.25],
                rot=[1.0, 0.0, 0.0, 0.0],
                joint_pos={
                    "openarm_left_joint1": -0.31204933,
                    "openarm_left_joint2": -0.42612678,
                    "openarm_left_joint3": 0.32234982,
                    "openarm_left_joint4": 0.43979153,
                    "openarm_left_joint5": -0.46879697,
                    "openarm_left_joint6": -0.25350952,
                    "openarm_left_joint7": -0.827409,
                    "openarm_right_joint1": 0.24847749,
                    "openarm_right_joint2": 0.00039903,
                    "openarm_right_joint3": -0.40817988,
                    "openarm_right_joint4": 0.6461343,
                    "openarm_right_joint5": 0.33295986,
                    "openarm_right_joint6": -0.24095318,
                    "openarm_right_joint7": 0.7177656,
                    "openarm_left_finger_joint1": 0.044,
                    "openarm_left_finger_joint2": 0.052,
                    "openarm_right_finger_joint1": 0.044,
                    "openarm_right_finger_joint2": 0.052,
                },
            ),
        )

        cup_usd = f"{OPENARM_ROOT_DIR}/usds/openarm_bimanual/cup.usd"
        bead_usd = f"{OPENARM_ROOT_DIR}/usds/openarm_bimanual/bead.usd"

        self.scene.object_source = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.1, 0.1, 0.05], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=cup_usd,
                scale=(0.8, 0.8, 1.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                ),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=100.0,
                    max_linear_velocity=100.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object/cup",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.1, 0.1, 0.05], rot=[1, 0, 0, 0]),
            spawn=None,
        )

        self.scene.object2_source = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Object2",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.1, -0.1, 0.05], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=cup_usd,
                scale=(0.8, 0.8, 1.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                ),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=100.0,
                    max_linear_velocity=100.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        self.scene.object2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object2/cup",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.1, -0.1, 0.05], rot=[1, 0, 0, 0]),
            spawn=None,
        )

        self.scene.bead = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bead",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.15, 0.0, 0.12], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=bead_usd,
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=100.0,
                    max_linear_velocity=100.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_joint[1-7]"],
            body_name="openarm_left_ee_tcp",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.3,
        )

        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_joint[1-7]"],
            body_name="openarm_right_ee_tcp",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.3,
        )
        self.actions.left_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_finger_joint.*"],
            scale=0.2,
            use_default_offset=True,
        )
        self.actions.right_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],
            scale=0.2,
            use_default_offset=True,
        )

        self.commands.left_object_pose.body_name = "openarm_left_ee_tcp"
        self.commands.right_object_pose.body_name = "openarm_right_ee_tcp"

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

        marker_cfg.prim_path = "/Visuals/LeftEEFrameTransformer"
        self.scene.left_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_left_ee_tcp",
                    name="left_end_effector",
                ),
            ],
        )
        marker_cfg.prim_path = "/Visuals/RightEEFrameTransformer"
        self.scene.right_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_right_ee_tcp",
                    name="right_end_effector",
                ),
            ],
        )

        marker_cfg.prim_path = "/Visuals/ObjectFrameTransformer"
        self.scene.object_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Object/cup/cup",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Object/cup/cup",
                    name="object_cup",
                ),
            ],
        )
        marker_cfg.prim_path = "/Visuals/Object2FrameTransformer"
        self.scene.object2_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Object2/cup/cup",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Object2/cup/cup",
                    name="object2_cup",
                ),
            ],
        )


@configclass
class Pouring3EnvCfg_RIGHT_ONLY(Pouring3EnvCfg):
    """Right-hand-only variant for debugging reach/goal behavior."""

    def __post_init__(self):
        super().__post_init__()
        # Freeze left arm/hand by zeroing action scale.
        self.actions.left_arm_action.scale = 0.0
        self.actions.left_hand_action.scale = 0.0
        # Disable left-side rewards/penalties to focus learning on right hand.
        self.rewards.left_reaching_object.weight = 0.0
        self.rewards.left_wrong_cup_penalty.weight = 0.0
        self.rewards.left_tcp_align_reward.weight = 0.0
        self.rewards.left_lifting_object.weight = 0.0
        self.rewards.left_object_goal_tracking.weight = 0.0
        self.rewards.left_object_goal_tracking_fine_grained.weight = 0.0
        self.rewards.left_hold_offset.weight = 0.0


@configclass
class Pouring3EnvCfg_PLAY(Pouring3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
