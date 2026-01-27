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

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg

from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
#from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

import math

##
# Scene definition
##


@configclass
class ReachIKSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # robots
    robot: ArticulationCfg = MISSING

    # target object
    object_source: AssetBaseCfg = MISSING
    object: RigidObjectCfg = MISSING
    object2_source: AssetBaseCfg = MISSING
    object2: RigidObjectCfg = MISSING

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.25, 0.0, 0.0], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

@configclass
class ReachIKCommandsCfg:
    """Command terms for the MDP."""

    left_object_pose = mdp.YAxisAlignedPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.45),
            pos_y=(0.15, 0.3),
            pos_z=(0.4, 0.6),
            roll=(-math.pi / 4, math.pi / 4),
            pitch=(math.pi / 4, 3 * math.pi / 4),
            yaw=(-math.pi, math.pi),
        ),
    )

    right_object_pose = mdp.YAxisAlignedPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.45),
            pos_y=(-0.3, -0.15),
            pos_z=(0.4, 0.6),
            roll=(-math.pi / 4, math.pi / 4),
            pitch=(math.pi / 4, 3 * math.pi / 4),
            yaw=(-math.pi, math.pi),
        ),
    )


@configclass
class ReachIKActionsCfg:
    """Action specifications for the MDP.

    DualHead를 위해 Left → Right 순서로 배치:
    [0:N] left_arm, [N:M] left_hand, [M:P] right_arm, [P:Q] right_hand
    """

    left_arm_action: ActionTerm = MISSING
    left_hand_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    right_hand_action: ActionTerm = MISSING


@configclass
class ReachIKObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_object_pose"}
        )
        target_object2_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "right_object_pose"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object")},
        )
        object2_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object2")},
        )
        object_obs = ObsTerm(
            func=mdp.object_obs,
            params={
                "left_eef_link_name": "openarm_left_hand",
                "right_eef_link_name": "openarm_right_hand",
            },
        )
        object2_obs = ObsTerm(
            func=mdp.object2_obs,
            params={
                "left_eef_link_name": "openarm_left_hand",
                "right_eef_link_name": "openarm_right_hand",
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class PolicyLowCfg(ObsGroup):
        """Observations for low-level skills (legacy shape)."""

        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_object_pose"}
        )
        target_object2_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "right_object_pose"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object")},
        )
        object2_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object2")},
        )
        object_obs = ObsTerm(
            func=mdp.object_obs,
            params={
                "left_eef_link_name": "openarm_left_hand",
                "right_eef_link_name": "openarm_right_hand",
            },
        )
        object2_obs = ObsTerm(
            func=mdp.object2_obs,
            params={
                "left_eef_link_name": "openarm_left_hand",
                "right_eef_link_name": "openarm_right_hand",
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy_low: PolicyLowCfg = PolicyLowCfg()

@configclass
class ReachIKEventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_from_command,
        mode="reset",
        params={
            "command_name": "left_object_pose",
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    reset_object2_position = EventTerm(
        func=mdp.reset_root_state_from_command,
        mode="reset",
        params={
            "command_name": "right_object_pose",
            "asset_cfg": SceneEntityCfg("object2"),
        },
    )

    follow_object_position = EventTerm(
        func=mdp.reset_root_state_from_command,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "command_name": "left_object_pose",
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    follow_object2_position = EventTerm(
        func=mdp.reset_root_state_from_command,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "command_name": "right_object_pose",
            "asset_cfg": SceneEntityCfg("object2"),
        },
    )


@configclass
class ReachIKRewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    left_end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.1,
            "command_name": "left_object_pose",
        },
    )

    right_end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.1,
            "command_name": "right_object_pose",
        },
    )

    # Orientation tracking - EE 축을 object 축과 일치시키기
    # To align hand +X with object +Z, use mdp.hand_x_align_object_z_reward.
    # If you want the old pose-tracking reward, swap func back to mdp.orientation_command_error_tanh.
    left_end_effector_orientation_tracking = RewTerm(
        func=mdp.hand_x_align_object_z_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="openarm_left_hand"),
            # Change this to "right_object_pose" if you intentionally want left hand aligned to right target.
            "command_name": "left_object_pose",
        },
    )

    right_end_effector_orientation_tracking = RewTerm(
        func=mdp.hand_x_align_object_z_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="openarm_right_hand"),
            "command_name": "right_object_pose",
        },
    )

    left_gripper_open = RewTerm(
        func=mdp.gripper_open_reward,
        weight=0.1,
        params={"eef_link_name": "openarm_left_hand"},
    )
    right_gripper_open = RewTerm(
        func=mdp.gripper_open_reward,
        weight=0.1,
        params={"eef_link_name": "openarm_right_hand"},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class ReachIKTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # joint_limit_near = DoneTerm(
    #     func=mdp.joint_limit_near_or_min_margin,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "margin_threshold": 0.001,
    #         "near_rate_threshold": 0.98,
    #         "warmup_steps": 1000,
    #     },
    # )


@configclass
class ReachIKCurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500},
    )


##
# Environment configuration
##


@configclass
class ReachIKEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachIKSceneCfg = ReachIKSceneCfg(num_envs=2048*1, env_spacing=2.5)
    # Basic settings
    observations: ReachIKObservationsCfg = ReachIKObservationsCfg()
    actions: ReachIKActionsCfg = ReachIKActionsCfg()
    commands: ReachIKCommandsCfg = ReachIKCommandsCfg()
    # MDP settings
    rewards: ReachIKRewardsCfg = ReachIKRewardsCfg()
    terminations: ReachIKTerminationsCfg = ReachIKTerminationsCfg()
    events: ReachIKEventCfg = ReachIKEventCfg()
    curriculum: ReachIKCurriculumCfg = ReachIKCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 24.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0
        self.sim.physx = PhysxCfg(
            solver_type=1,  # TGS
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            # increase buffers to prevent overflow errors
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=8,
            gpu_collision_stack_size=2**24,  # unsigned int 범위 내 값
        )
