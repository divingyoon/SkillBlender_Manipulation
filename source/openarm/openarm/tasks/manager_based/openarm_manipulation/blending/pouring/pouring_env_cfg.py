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
import math

import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from openarm.tasks.manager_based.openarm_manipulation.bimanual.grasp_2g import mdp as grasp2g_mdp


@configclass
class PouringSceneCfg(InteractiveSceneCfg):
    """Scene with a bimanual robot, table, and a cube for handover."""

    robot: ArticulationCfg = MISSING

    object_source: AssetBaseCfg = MISSING
    object: RigidObjectCfg = MISSING
    object2_source: AssetBaseCfg = MISSING
    object2: RigidObjectCfg = MISSING
    bead: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.25, 0.0, 0.0], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    left_object_pose = mdp.PhaseSwitchPoseCommandCfg(
        asset_name="robot",
        body_name="openarm_left_ee_tcp",
        source_asset_cfg=SceneEntityCfg("object"),
        target_asset_cfg=SceneEntityCfg("object"),
        resampling_time_range=(4.0, 4.0),
        pre_offset=(0.0, 0.0, 0.03),
        post_offset=(0.0, 0.0, 0.12),
        switch_phase=3,
        phase_source="group",
        ranges=mdp.PhaseSwitchPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.05),
            pos_y=(0.05, 0.15),
            pos_z=(0.1, 0.2),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        debug_vis=True,
    )

    right_object_pose = mdp.PhaseSwitchPoseCommandCfg(
        asset_name="robot",
        body_name="openarm_right_ee_tcp",
        source_asset_cfg=SceneEntityCfg("object2"),
        target_asset_cfg=SceneEntityCfg("object2"),
        resampling_time_range=(4.0, 4.0),
        pre_offset=(0.0, 0.0, 0.03),
        post_offset=(0.0, 0.0, 0.12),
        switch_phase=3,
        phase_source="group",
        ranges=mdp.PhaseSwitchPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.05),
            pos_y=(-0.15, -0.05),
            pos_z=(0.1, 0.2),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    left_hand_action: ActionTerm = MISSING
    right_hand_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_object_pose"}
        )
        target_object2_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "right_object_pose"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
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
                "left_eef_link_name": "openarm_left_ee_tcp",
                "right_eef_link_name": "openarm_right_ee_tcp",
                "command_name": "left_object_pose",
            },
        )
        object2_obs = ObsTerm(
            func=mdp.object2_obs,
            params={
                "left_eef_link_name": "openarm_left_ee_tcp",
                "right_eef_link_name": "openarm_right_ee_tcp",
                "command_name": "right_object_pose",
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

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
                "left_eef_link_name": "openarm_left_ee_tcp",
                "right_eef_link_name": "openarm_right_ee_tcp",
                "command_name": "left_object_pose",
            },
        )
        object2_obs = ObsTerm(
            func=mdp.object2_obs,
            params={
                "left_eef_link_name": "openarm_left_ee_tcp",
                "right_eef_link_name": "openarm_right_ee_tcp",
                "command_name": "right_object_pose",
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy_low: PolicyLowCfg = PolicyLowCfg()

    @configclass
    class HighLevelCfg(ObsGroup):
        """Additional observations for high-level pouring policy."""

        cup_pair = ObsTerm(
            func=mdp.cup_pair_compact_obs,
            params={"source_name": "object", "target_name": "object2"},
        )
        bead = ObsTerm(
            func=mdp.bead_to_target_obs,
            params={"bead_name": "bead", "target_name": "object2"},
        )
        phase_left = ObsTerm(func=mdp.pour_phase_left)
        phase_right = ObsTerm(func=mdp.pour_phase_right)
        phase_group = ObsTerm(func=mdp.pour_phase_group)
        # Optional per-hand high-level obs (kept for future separation)
        # left_object_obs = ObsTerm(
        #     func=mdp.object_position_in_robot_root_frame,
        #     params={"object_cfg": SceneEntityCfg("object")},
        # )
        # right_object_obs = ObsTerm(
        #     func=mdp.object_position_in_robot_root_frame,
        #     params={"object_cfg": SceneEntityCfg("object2")},
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    high_level: HighLevelCfg = HighLevelCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.3, 0.3), "y": (0.1, 0.1), "z": (0.0, 0.0),
                "yaw": (-math.pi / 2, -math.pi / 2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    reset_object2_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.3, 0.3), "y": (-0.1, -0.1), "z": (0.0, 0.0),
                "yaw": (-math.pi / 2, -math.pi / 2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object2"),
        },
    )
    reset_bead_position = EventTerm(
        func=mdp.reset_bead_in_cup,
        mode="reset",
        params={
            "cup_name": "object",
            "bead_name": "bead",
            "offset": (0.0, 0.0, 0.05),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    phase_left = RewTerm(
        func=mdp.phase_left_value,
        weight=0.0,
    )
    phase_right = RewTerm(
        func=mdp.phase_right_value,
        weight=0.0,
    )
    phase_shared = RewTerm(
        func=mdp.phase_shared_value,
        weight=0.0,
    )
    left_reaching_object = RewTerm(
        func=grasp2g_mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("object"), "ee_frame_cfg": SceneEntityCfg("left_ee_frame")},
        weight=2.0,
    )
    right_reaching_object = RewTerm(
        func=grasp2g_mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("object2"), "ee_frame_cfg": SceneEntityCfg("right_ee_frame")},
        weight=3.0,
    )
    left_wrong_cup_penalty = RewTerm(
        func=mdp.phase_wrong_cup_penalty,
        weight=-1.0,
        params={
            "eef_link_name": "openarm_left_ee_tcp",
            "object_cfg": SceneEntityCfg("object2"),
        },
    )
    right_wrong_cup_penalty = RewTerm(
        func=mdp.phase_wrong_cup_penalty,
        weight=-0.2,
        params={
            "eef_link_name": "openarm_right_ee_tcp",
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    left_tcp_align_reward = RewTerm(
        func=mdp.phase_tcp_x_axis_alignment,
        weight=0.0,
        params={
            "eef_link_name": "openarm_left_ee_tcp",
            "object_cfg": SceneEntityCfg("object"),
            "hand": "left",
        },
    )
    right_tcp_align_reward = RewTerm(
        func=mdp.phase_tcp_x_axis_alignment,
        weight=0.0,
        params={
            "eef_link_name": "openarm_right_ee_tcp",
            "object_cfg": SceneEntityCfg("object2"),
            "hand": "right",
        },
    )

    left_lifting_object = RewTerm(
        func=grasp2g_mdp.phase_lift_reward,
        params={
            "lift_height": 0.1,
            "object_cfg": SceneEntityCfg("object"),
            "phase_weights": [0.0, 0.0, 1.0, 1.0],
            "phase_params": {
                "eef_link_name": "openarm_left_ee_tcp",
                "lift_height": 0.1,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=15.0,
    )
    right_lifting_object = RewTerm(
        func=grasp2g_mdp.phase_lift_reward,
        params={
            "lift_height": 0.1,
            "object_cfg": SceneEntityCfg("object2"),
            "phase_weights": [0.0, 0.0, 1.0, 1.0],
            "phase_params": {
                "eef_link_name": "openarm_right_ee_tcp",
                "lift_height": 0.1,
                "reach_distance": 0.05,
                "align_threshold": 0.0,
                "grasp_distance": 0.02,
                "close_threshold": 0.6,
                "hold_duration": 2.0,
            },
        },
        weight=15.0,
    )

    left_object_goal_tracking = RewTerm(
        func=mdp.phase_object_goal_distance_with_ee,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "left_object_pose",
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "reach_std": 0.1,
            "hand": "left",
            "min_phase": 2,
        },
        weight=16.0,
    )
    right_object_goal_tracking = RewTerm(
        func=mdp.phase_object_goal_distance_with_ee,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "right_object_pose",
            "object_cfg": SceneEntityCfg("object2"),
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "reach_std": 0.1,
            "hand": "right",
            "min_phase": 2,
        },
        weight=16.0,
    )

    left_object_goal_tracking_fine_grained = RewTerm(
        func=mdp.phase_object_goal_distance_with_ee,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "left_object_pose",
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "reach_std": 0.1,
            "hand": "left",
            "min_phase": 2,
        },
        weight=5.0,
    )
    right_object_goal_tracking_fine_grained = RewTerm(
        func=mdp.phase_object_goal_distance_with_ee,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "right_object_pose",
            "object_cfg": SceneEntityCfg("object2"),
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "reach_std": 0.1,
            "hand": "right",
            "min_phase": 2,
        },
        weight=5.0,
    )
    left_hold_offset = RewTerm(
        func=mdp.hold_at_offset_reward,
        params={
            "eef_link_name": "openarm_left_ee_tcp",
            "object_cfg": SceneEntityCfg("object"),
            "offset_z": 0.2,
            "grasp_distance": 0.02,
            "close_threshold": 0.6,
            "std": 0.05,
            "hold_duration": 2.0,
            "hold_distance": 0.05,
        },
        weight=5.0,
    )
    right_hold_offset = RewTerm(
        func=mdp.hold_at_offset_reward,
        params={
            "eef_link_name": "openarm_right_ee_tcp",
            "object_cfg": SceneEntityCfg("object2"),
            "offset_z": 0.2,
            "grasp_distance": 0.02,
            "close_threshold": 0.6,
            "std": 0.05,
            "hold_duration": 2.0,
            "hold_distance": 0.05,
        },
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    phase_tracker = RewTerm(
        func=mdp.phase_reach_reward,
        weight=0.0,
        params={
            "std": 0.1,
            "eef_link_name": "openarm_left_ee_tcp",
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    bead_spill = DoneTerm(
        func=mdp.bead_spill,
        params={
            "bead_name": "bead",
            "target_name": "object2",
            "min_height_offset": -0.04,
            "xy_radius": 0.08,
        },
    )

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )
    object2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object2")},
    )
    object_tipped = DoneTerm(
        func=mdp.cup_tipped,
        params={"object_name": "object", "min_upright_dot": 0.5},
    )
    object2_tipped = DoneTerm(
        func=mdp.cup_tipped,
        params={"object_name": "object2", "min_upright_dot": 0.5},
    )
    object_out_of_reach = DoneTerm(
        func=mdp.object_out_of_reach,
        params={"object_cfg": SceneEntityCfg("object"), "max_xy": 0.7, "max_z": 0.4, "min_z": -0.05},
    )
    object2_out_of_reach = DoneTerm(
        func=mdp.object_out_of_reach,
        params={"object_cfg": SceneEntityCfg("object2"), "max_xy": 0.7, "max_z": 0.4, "min_z": -0.05},
    )


@configclass
class PouringBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the bimanual pouring blending environment."""

    scene: PouringSceneCfg = PouringSceneCfg(num_envs=2048*1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    commands: CommandsCfg = CommandsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.physx = PhysxCfg(
            solver_type=1,  # TGS
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            enable_ccd=True,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            # increase buffers to prevent overflow errors
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=8,
            gpu_collision_stack_size=1000000000,
        )
