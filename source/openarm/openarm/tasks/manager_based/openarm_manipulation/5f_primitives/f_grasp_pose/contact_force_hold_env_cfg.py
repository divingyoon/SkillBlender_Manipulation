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

"""Primitive B: Contact Force Hold - Environment Configuration.

This is a task-agnostic, goal-conditioned motor skill for maintaining stable grasps.
NO GWS/epsilon, NO task success metrics. Margin-based holding only.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

from ..common.robot_cfg import (
    LEFT_HAND_JOINTS,
    RIGHT_HAND_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    LEFT_EE_FRAME,
    RIGHT_EE_FRAME,
    LEFT_GRASP_FRAME,
    LEFT_CONTACT_LINKS,
    RIGHT_CONTACT_LINKS,
    CONTROL_RATE_HZ,
)

LEFT_CONTACT_SENSOR_NAMES = ["left_contact_sensor"] + [
    f"left_contact_sensor_{idx}" for idx in range(2, len(LEFT_CONTACT_LINKS) + 1)
]
RIGHT_CONTACT_SENSOR_NAMES = ["right_contact_sensor"] + [
    f"right_contact_sensor_{idx}" for idx in range(2, len(RIGHT_CONTACT_LINKS) + 1)
]

##
# Scene definition
##


@configclass
class ContactForceHoldSceneCfg(InteractiveSceneCfg):
    """Scene configuration for contact force hold primitive.

    Tabletop grasp scenario with graspable object and contact sensors.
    """

    # Robot - will be populated by agent env cfg
    robot: ArticulationCfg = MISSING

    # Graspable object - will be populated by agent env cfg
    object: RigidObjectCfg = MISSING

    # Contact sensors for fingertips - will be populated by agent env cfg
    left_contact_sensor: ContactSensorCfg = MISSING
    right_contact_sensor: ContactSensorCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.80, 0.0, 0.0),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP.

    Generates grip margin target bands.
    """

    # Grip force target band (low, high) in normalized units
    # TODO: Implement GripMarginCommandCfg in mdp
    grip_margin = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),  # Keep constant during episode
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.7),  # Re-purpose as force_low
            lin_vel_y=(0.5, 0.9),  # Re-purpose as force_high
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Joint position targets for hands (and optional wrist).
    """

    left_hand_action: ActionTerm = MISSING
    right_hand_action: ActionTerm = MISSING
    left_arm_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.

        State observations:
        - Hand joint positions/velocities
        - Contact flags per fingertip
        - Normal force proxy per fingertip
        - Slip velocity proxy per fingertip

        Goal observations:
        - Grip margin target band
        """

        # Left hand joint state
        left_arm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        left_arm_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        left_hand_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_HAND_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        left_hand_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_HAND_JOINTS)},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # Right hand joint state
        right_hand_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        right_hand_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS)},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        right_arm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        right_arm_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # Contact observations (custom functions)
        left_contact_flags = ObsTerm(
            func=mdp.contact_flags_multi,
            params={"sensor_names": LEFT_CONTACT_SENSOR_NAMES},
        )

        right_contact_flags = ObsTerm(
            func=mdp.contact_flags_multi,
            params={"sensor_names": RIGHT_CONTACT_SENSOR_NAMES},
        )

        left_normal_forces = ObsTerm(
            func=mdp.normal_force_magnitude_multi,
            params={"sensor_names": LEFT_CONTACT_SENSOR_NAMES},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        right_normal_forces = ObsTerm(
            func=mdp.normal_force_magnitude_multi,
            params={"sensor_names": RIGHT_CONTACT_SENSOR_NAMES},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        object_cross_section_size = ObsTerm(
            func=mdp.object_cross_section_size,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "cylinder_radius": 0.05,
                "box_size_xy": (0.08, 0.08),
            },
        )

        # Goal: grip margin target
        grip_target = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "grip_margin"},
        )

        # Previous actions
        left_actions = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "left_hand_action"},
        )

        right_actions = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "right_hand_action"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Robot hand material (fixed rubber-like friction).
    hand_physics = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_CONTACT_LINKS + RIGHT_CONTACT_LINKS),
            "static_friction_range": (1.2, 1.2),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    # Object physics randomization (curriculum-controlled)
    object_physics = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (1.2, 1.2),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 8,
        },
    )

    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "mass_distribution_params": (0.2, 0.2),
            "operation": "abs",
            "recompute_inertia": True,
        },
    )

    object_com = EventTerm(
        func=mdp.randomize_object_com,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object", body_names=".*"),
            "com_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )

    object_scale = EventTerm(
        func=mdp.randomize_object_shape_scale,
        mode="prestartup",
        params={
            "object_cfg": SceneEntityCfg("object"),
            "cylinder_scale_xy_range": (1.0, 1.1),
            "cube_scale_xy_range": (1.0, 1.0625),
        },
    )

    # Reset robot/object to contact-hold states (B1/B2 curriculum)
    reset_contact_state = EventTerm(
        func=mdp.reset_contact_force_hold_state,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object"),
            "robot_cfg": SceneEntityCfg("robot"),
            "left_hand_joint_names": LEFT_HAND_JOINTS,
            "right_hand_joint_names": RIGHT_HAND_JOINTS,
            "left_arm_joint_names": LEFT_ARM_JOINTS,
            "right_arm_joint_names": RIGHT_ARM_JOINTS,
            "arm_joint_targets": {
                "openarm_left_joint1": 0.0,
                "openarm_left_joint2": -0.1,
                "openarm_left_joint3": 0.0,
                "openarm_left_joint4": 0.7745,
                "openarm_left_joint5": 0.0,
                "openarm_left_joint6": 0.1,
                "openarm_left_joint7": -0.7,
                "openarm_right_joint1": 0.2,
                "openarm_right_joint2": 0.0,
                "openarm_right_joint3": 0.0,
                "openarm_right_joint4": 2.1708,
                "openarm_right_joint5": 0.0,
                "openarm_right_joint6": 0.1,
                "openarm_right_joint7": 0.7,
            },
            "object_pose_b1": {"x": (-0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "object_pose_b2": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "grasp_frame_name": LEFT_GRASP_FRAME,
            "object_offset_b1": {"x": (0.10, 0.10), "y": (0.005, 0.005), "z": (0.00, 0.00)},
            "object_offset_b2": {"x": (0.10, 0.10), "y": (0.005, 0.005), "z": (0.00, 0.00)},
            "b1_prob_init": 0.9,
            "b1_prob_final": 0.3,
            "b1_prob_steps": 10000,
            "hand_bias_b1": 0.5,
            "hand_bias_b2": 0.35,
            "hand_extra_bias_b1": 0.13,
            "hand_extra_bias_b2": 0.13,
            "hand_extra_bias_lj_dg_4": 0.08,
            "left_hand_offset_targets": {
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
            "left_hand_offset_noise": 0.02,
            "hand_noise": 0.0,
            "arm_noise": 0.0,
            "active_hand": "left",
            "left_arm_fixed_joints": ["openarm_left_joint1", "openarm_left_joint2"],
            "left_arm_fixed_noise_deg": 0.0,
        },
    )


    # Micro disturbances (late curriculum)
    micro_disturbances = EventTerm(
        func=mdp.apply_micro_disturbances,
        mode="interval",
        interval_range_s=(0.4, 0.8),
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_FRAME, RIGHT_EE_FRAME]),
            "force_range_initial": (0.0, 0.0),
            "force_range_final": (0.5, 1.0),
            "torque_range_initial": (0.0, 0.0),
            "torque_range_final": (0.01, 0.03),
            "steps": 15000,
        },
    )

    # Debug: visualize left fingertip contact force vectors (env0 only).
    debug_tip_force_vectors = EventTerm(
        func=mdp.debug_tip_force_vectors,
        mode="interval",
        interval_range_s=(0.01, 0.01),
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_names": LEFT_CONTACT_SENSOR_NAMES,
            "link_names": LEFT_CONTACT_LINKS,
            "env_id": 0,
            "max_force": 50.0,
            "enabled": True,
            "log_force_vectors": True,
            "log_every": 1,
        },
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Task-agnostic grasp maintenance rewards. NO GWS/epsilon or task success.

    Initial weights (reasonable defaults):
    - Contact persistence: weight=0.5 (reward for maintaining contacts)
    - Force spike penalty: weight=-0.2 (penalty for impact spikes)
    - Overgrip penalty: weight=-0.1 (penalty for excessive force)
    - Action rate: weight=-0.001 (smoothness)
    - Safety: weight=-0.1 (joint limits, self-collision)
    """

    # === Contact Persistence ===
    # left_missing_contact_penalty = RewTerm(
    #     func=mdp.missing_contact_penalty,
    #     weight=-0.1,
    #     params={
    #         "sensor_names": ["left_contact_sensor_4", "left_contact_sensor_5"],
    #         "contact_threshold": 0.05,
    #     },
    # )

    # any_contact_penalty = RewTerm(
    #     func=mdp.any_contact_penalty_multi,
    #     weight=-0.001,
    #     params={
    #         "sensor_names": LEFT_CONTACT_SENSOR_NAMES + RIGHT_CONTACT_SENSOR_NAMES,
    #         "contact_threshold": 0.1,
    #     },
    # )

    left_link4_spacing_reward = RewTerm(
        func=mdp.finger_link_spacing_reward,
        weight=0.1,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=LEFT_CONTACT_LINKS),
            "distance_scale": 0.05,
        },
    )

    left_link4_spacing_object_penalty = RewTerm(
        func=mdp.finger_link_spacing_object_size_penalty,
        weight=-0.1,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=LEFT_CONTACT_LINKS),
            "object_cfg": SceneEntityCfg("object"),
            "cylinder_radius": 0.05,
            "box_size_xy": (0.08, 0.08),
        },
    )

    # # === Force Spike Penalty ===
    # left_force_spike = RewTerm(
    #     func=mdp.force_spike_penalty_multi,
    #     weight=-0.2,
    #     params={
    #         "sensor_names": LEFT_CONTACT_SENSOR_NAMES,
    #         "spike_threshold": 10.0,
    #         "contact_threshold": 0.05,
    #     },
    # )

    # right_force_spike = RewTerm(
    #     func=mdp.force_spike_penalty_multi,
    #     weight=0.0,
    #     params={
    #         "sensor_names": RIGHT_CONTACT_SENSOR_NAMES,
    #         "spike_threshold": 10.0,
    #         "contact_threshold": 0.05,
    #     },
    # )

    # # === Overgrip Penalty ===
    # left_overgrip = RewTerm(
    #     func=mdp.overgrip_penalty_multi,
    #     weight=-0.08,
    #     params={
    #         "sensor_names": LEFT_CONTACT_SENSOR_NAMES,
    #         "max_force": 15.0,
    #         "contact_threshold": 0.05,
    #     },
    # )

    # right_overgrip = RewTerm(
    #     func=mdp.overgrip_penalty_multi,
    #     weight=0.0,
    #     params={
    #         "sensor_names": RIGHT_CONTACT_SENSOR_NAMES,
    #         "max_force": 15.0,
    #         "contact_threshold": 0.05,
    #     },
    # )

    termination_failure_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-0.5,
        params={"term_keys": ["object_dropped", "object_tipped"]},
    )

    # === Smoothness Penalties ===
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.001,
    )

    left_hand_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_HAND_JOINTS)},
    )

    right_hand_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_JOINTS)},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Timeout after 8 seconds
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Object dropped (height below threshold)
    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": -0.1,
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    # Sustained contact loss (all contacts lost)
    # contact_lost = DoneTerm(
    #     func=mdp.all_contacts_lost_multi,
    #     params={
    #         "left_sensor_names": LEFT_CONTACT_SENSOR_NAMES,
    #         "right_sensor_names": [],
    #         "num_steps": 20,
    #     },
    # )

    # Object tipped over (fallen)
    object_tipped = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": 0.9,
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.

    Stage 1: Hold stationary object
    Stage 2: Hold with external disturbances
    Stage 3: Randomize mass/friction/CoM
    """

    pushout_penalty_ramp = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.left_pushout_velocity_penalty.weight",
            "modify_fn": mdp.delayed_linear_interpolate_fn,
            "modify_params": {
                "initial_value": 0.0,
                "final_value": -0.1,
                "delay_steps": 30000,
                "num_steps": 30000,
            },
        },
    )

    termination_failure_penalty_ramp = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.termination_failure_penalty.weight",
            "modify_fn": mdp.delayed_linear_interpolate_fn,
            "modify_params": {
                "initial_value": 0.0,
                "final_value": -0.5,
                "delay_steps": 20000,
                "num_steps": 20000,
            },
        },
    )

    # Ramp object physics randomization
    object_static_friction = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.object_physics.params.static_friction_range",
            "modify_fn": mdp.delayed_linear_interpolate_fn,
            "modify_params": {
                "initial_value": (1.2, 1.2),
                "final_value": (0.8, 1.2),
                "delay_steps": 100000,
                "num_steps": 100000,
            },
        },
    )

    object_dynamic_friction = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.object_physics.params.dynamic_friction_range",
            "modify_fn": mdp.delayed_linear_interpolate_fn,
            "modify_params": {
                "initial_value": (1.0, 1.0),
                "final_value": (0.7, 1.0),
                "delay_steps": 100000,
                "num_steps": 100000,
            },
        },
    )

    object_mass_range = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.object_mass.params.mass_distribution_params",
            "modify_fn": mdp.delayed_linear_interpolate_fn,
            "modify_params": {
                "initial_value": (0.2, 0.2),
                "final_value": (0.25, 0.4),
                "delay_steps": 100000,
                "num_steps": 100000,
            },
        },
    )

    object_com_range = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.object_com.params.com_range",
            "modify_fn": mdp.linear_interpolate_fn,
            "modify_params": {
                "initial_value": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                "final_value": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
                "num_steps": 10000,
            },
        },
    )


##
# Environment configuration
##


@configclass
class ContactForceHoldEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the contact force hold environment.

    This is the abstract base configuration. Derived configs should set:
    - scene.robot: The robot articulation config
    - scene.object: The graspable object config
    - scene.*_contact_sensor: Contact sensors for fingertips
    """

    # Scene settings
    scene: ContactForceHoldSceneCfg = ContactForceHoldSceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 2  # 50 Hz control at 100 Hz sim
        self.episode_length_s = 8.0  # 8 second episodes

        # Simulation settings
        self.sim.dt = 1.0 / 100.0  # 100 Hz simulation
        self.sim.render_interval = self.decimation

        # Physics settings for contact
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        # Increase GPU buffer sizes for large contact counts.
        self.sim.physx.gpu_max_rigid_patch_count = 4_000_000
        self.sim.physx.gpu_max_rigid_contact_count = 16_000_000
        self.sim.physx.gpu_found_lost_pairs_capacity = 4_000_000
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 8_000_000
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 256 * 1024
        self.sim.physx.gpu_collision_stack_size = 128 * 1024 * 1024
        self.sim.physx.gpu_heap_capacity = 128 * 1024 * 1024
        self.sim.physx.gpu_temp_buffer_capacity = 32 * 1024 * 1024

        # Viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.lookat = (0.5, 0.0, 0.2)

        # Multi-asset spawns per env require disabling physics replication.
        self.scene.replicate_physics = False
