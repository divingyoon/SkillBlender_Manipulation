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


@configclass
class PickAndPlaceBlendingSceneCfg(InteractiveSceneCfg):
    """Scene with a bimanual robot, table, and a cube for handover."""

    robot: ArticulationCfg = MISSING

    object: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.25, 0.0], rot=[0.707, 0, 0, 0.707]),
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

    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.3),
            pos_y=(0.15, 0.25),
            pos_z=(0.3, 0.5),
            roll=(0.0, 0.0),
            pitch=(1.5707963267948966, 1.5707963267948966),
            yaw=(-1.5707963267948966, -1.5707963267948966),
        ),
    )

    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.3),
            pos_y=(-0.25, -0.15),
            pos_z=(0.3, 0.5),
            roll=(0.0, 0.0),
            pitch=(1.5707963267948966, 1.5707963267948966),
            yaw=(1.5707963267948966, 1.5707963267948966),
        ),
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

        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "openarm_left_joint1",
                        "openarm_left_joint2",
                        "openarm_left_joint3",
                        "openarm_left_joint4",
                        "openarm_left_joint5",
                        "openarm_left_joint6",
                        "openarm_left_joint7",
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "openarm_right_joint1",
                        "openarm_right_joint2",
                        "openarm_right_joint3",
                        "openarm_right_joint4",
                        "openarm_right_joint5",
                        "openarm_right_joint6",
                        "openarm_right_joint7",
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "openarm_left_joint1",
                        "openarm_left_joint2",
                        "openarm_left_joint3",
                        "openarm_left_joint4",
                        "openarm_left_joint5",
                        "openarm_left_joint6",
                        "openarm_left_joint7",
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "openarm_right_joint1",
                        "openarm_right_joint2",
                        "openarm_right_joint3",
                        "openarm_right_joint4",
                        "openarm_right_joint5",
                        "openarm_right_joint6",
                        "openarm_right_joint7",
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_hand_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_finger_joint.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_hand_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_hand_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_finger_joint.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_hand_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "left_ee_pose"},
        )
        right_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "right_ee_pose"},
        )
        object = ObsTerm(
            func=mdp.object_obs,
            params={
                "left_eef_link_name": MISSING,
                "right_eef_link_name": MISSING,
            },
        )
        object2 = ObsTerm(
            func=mdp.object2_obs,
            params={
                "left_eef_link_name": MISSING,
                "right_eef_link_name": MISSING,
            },
        )
        left_arm_actions = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        right_arm_actions = ObsTerm(func=mdp.last_action, params={"action_name": "right_arm_action"})
        left_hand_actions = ObsTerm(func=mdp.last_action, params={"action_name": "left_hand_action"})
        right_hand_actions = ObsTerm(func=mdp.last_action, params={"action_name": "right_hand_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.0),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    left_eef_to_object_distance = RewTerm(
        func=mdp.eef_to_object_distance,
        weight=1.0,
        params={"std": 0.15, "eef_link_name": MISSING, "object_cfg": SceneEntityCfg("object")},
    )
    right_eef_to_object_distance = RewTerm(
        func=mdp.eef_to_object_distance,
        weight=1.0,
        params={"std": 0.15, "eef_link_name": MISSING, "object_cfg": SceneEntityCfg("object")},
    )
    left_grasp_reward = RewTerm(
        func=mdp.grasp_reward,
        weight=10.0,
        params={"eef_link_name": "openarm_left_hand", "object_cfg": SceneEntityCfg("object")},
    )
    right_grasp_reward = RewTerm(
        func=mdp.grasp_reward,
        weight=10.0,
        params={"eef_link_name": "openarm_right_hand", "object_cfg": SceneEntityCfg("object")},
    )
    left_lift_reward = RewTerm(
        func=mdp.object_is_lifted,
        weight=5.0,
        params={"minimal_height": 0.05, "object_cfg": SceneEntityCfg("object")},
    )
    left_hold_reward = RewTerm(
        func=mdp.object_is_held,
        weight=10.0,
        params={"minimal_height": 0.05, "hold_duration": 2.0, "object_cfg": SceneEntityCfg("object")},
    )
    handover_reward = RewTerm(
        func=mdp.handover_success,
        weight=10.0,
        params={"eef_link_name_right": "openarm_right_hand", "object_cfg": SceneEntityCfg("object")},
    )
    place_reward = RewTerm(
        func=mdp.object_at_target,
        weight=20.0,
        params={"target_pos": (0.2, -0.2, 0.05), "object_cfg": SceneEntityCfg("object")},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "openarm_left_joint1",
                    "openarm_left_joint2",
                    "openarm_left_joint3",
                    "openarm_left_joint4",
                    "openarm_left_joint5",
                    "openarm_left_joint6",
                    "openarm_left_joint7",
                ],
            )
        },
    )
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "openarm_right_joint1",
                    "openarm_right_joint2",
                    "openarm_right_joint3",
                    "openarm_right_joint4",
                    "openarm_right_joint5",
                    "openarm_right_joint6",
                    "openarm_right_joint7",
                ],
            )
        },
    )
    left_hand_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.00005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_finger_joint.*"])},
    )
    right_hand_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.00005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.00, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class PickAndPlaceBlendingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the bimanual pick-and-place blending environment."""

    scene: PickAndPlaceBlendingSceneCfg = PickAndPlaceBlendingSceneCfg(num_envs=20480, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    commands: CommandsCfg = CommandsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.5, 3.5, 3.5)
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
            gpu_collision_stack_size=640000,
        )