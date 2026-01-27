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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass


from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR
from openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import (
    OPEN_ARM_HIGH_PD_CFG,
)
from .. import mdp
from ..TransferIK_env_cfg import TransferIKEnvCfg

@configclass
class TransferIKJointPosEnvCfg(TransferIKEnvCfg):
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
                    "openarm_left_joint1": -0.5,
                    "openarm_left_joint2": -0.5,
                    "openarm_left_joint3": 0.6,
                    "openarm_left_joint4": 0.7,
                    "openarm_left_joint5": 0.0,
                    "openarm_left_joint6": 0.0,
                    "openarm_left_joint7": -1.0,
                    "openarm_right_joint1": 0.5,
                    "openarm_right_joint2": 0.5,
                    "openarm_right_joint3": -0.6,
                    "openarm_right_joint4": 0.7,
                    "openarm_right_joint5": 0.0,
                    "openarm_right_joint6": 0.0,
                    "openarm_right_joint7": 1.0,
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


        # DualHead를 위해 Left → Right 순서로 배치
        # [0:7] left_arm, [7:9] left_hand, [9:16] right_arm, [16:18] right_hand
        # dof_split_index = 9 (left: 0~8, right: 9~17)
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_left_joint*",
            ],
            scale=0.2,
            use_default_offset=True,
        )
        self.actions.left_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_finger_joint.*"],
            scale=0.15,
            use_default_offset=True,
        )

        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_right_joint*",
            ],
            scale=0.2,
            use_default_offset=True,
        )
        self.actions.right_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],
            scale=0.15,
            use_default_offset=True,
        )

        # Commands should target the palm frames.
        self.commands.left_object_pose.body_name = "openarm_left_hand"
        self.commands.right_object_pose.body_name = "openarm_right_hand"
        
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        
        # End-effector frame for reach/lift rewards (prim_path must be a rigid body)
        marker_cfg.prim_path = "/Visuals/LeftEEFrameTransformer"
        self.scene.left_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand",
                    name="left_end_effector",
                ),
            ],
        )
        marker_cfg.prim_path = "/Visuals/RightEEFrameTransformer"
        self.scene.right_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand",
                    name="right_end_effector",
                ),
            ],
        )


@configclass
class TransferIKJointPosEnvCfg_PLAY(TransferIKJointPosEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
