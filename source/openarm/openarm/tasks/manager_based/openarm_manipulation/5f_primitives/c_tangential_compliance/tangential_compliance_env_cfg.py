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

"""Primitive C: Tangential Compliance - Environment Configuration.

Task-agnostic controlled slip. Does NOT target final object pose.
Local compliance only.
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
    LEFT_CONTACT_LINKS,
    RIGHT_CONTACT_LINKS,
)


@configclass
class TangentialComplianceSceneCfg(InteractiveSceneCfg):
    """Scene configuration for tangential compliance primitive."""

    robot: ArticulationCfg = MISSING
    object: RigidObjectCfg = MISSING
    left_contact_sensor: ContactSensorCfg = MISSING
    right_contact_sensor: ContactSensorCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
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
    """Command terms: allowed slip direction and compliance scale."""

    # Slip direction command (unit vector in tangent plane)
    # Using velocity command format: (dir_x, dir_y, dir_z, compliance_scale)
    slip_direction = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),   # direction x
            lin_vel_y=(-1.0, 1.0),   # direction y
            # lin_vel_z=(0.0, 0.0),    # direction z (typically 0 for tangent plane)
            ang_vel_z=(0.5, 2.0),    # compliance scale
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications."""

    left_hand_action: ActionTerm = MISSING
    right_hand_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

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

        left_contact_flags = ObsTerm(
            func=mdp.contact_flags,
            params={"sensor_cfg": SceneEntityCfg("left_contact_sensor")},
        )

        right_contact_flags = ObsTerm(
            func=mdp.contact_flags,
            params={"sensor_cfg": SceneEntityCfg("right_contact_sensor")},
        )

        left_normal_forces = ObsTerm(
            func=mdp.normal_force_magnitude,
            params={"sensor_cfg": SceneEntityCfg("left_contact_sensor")},
        )

        right_normal_forces = ObsTerm(
            func=mdp.normal_force_magnitude,
            params={"sensor_cfg": SceneEntityCfg("right_contact_sensor")},
        )

        # Slip velocity vector (not just magnitude)
        left_slip_vector = ObsTerm(
            func=mdp.slip_velocity_vector,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=LEFT_CONTACT_LINKS),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        right_slip_vector = ObsTerm(
            func=mdp.slip_velocity_vector,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=RIGHT_CONTACT_LINKS),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        # Goal: allowed slip direction
        slip_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "slip_direction"},
        )

        left_actions = ObsTerm(func=mdp.last_action, params={"action_name": "left_hand_action"})
        right_actions = ObsTerm(func=mdp.last_action, params={"action_name": "right_hand_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event configuration."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.01)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 16,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms.

    Weights:
    - Keep contact: 0.4
    - Non-allowed slip penalty: -0.4
    - Allowed slip band: 0.3
    - Force stability: -0.15
    - Action smoothness: -0.001
    """

    # Keep contact reward
    keep_contact = RewTerm(
        func=mdp.contact_persistence_reward,
        weight=0.4,
        params={"sensor_cfg": SceneEntityCfg("left_contact_sensor"), "min_contacts": 3},
    )

    # Non-allowed slip penalty (perpendicular component)
    left_nonallowed_slip = RewTerm(
        func=mdp.nonallowed_slip_penalty,
        weight=-0.4,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=LEFT_CONTACT_LINKS),
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "slip_direction",
        },
    )

    right_nonallowed_slip = RewTerm(
        func=mdp.nonallowed_slip_penalty,
        weight=-0.4,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=RIGHT_CONTACT_LINKS),
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "slip_direction",
        },
    )

    # Allowed slip band reward
    left_slip_band = RewTerm(
        func=mdp.slip_in_allowed_band,
        weight=0.3,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=LEFT_CONTACT_LINKS),
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "slip_direction",
            "min_slip": 0.0,
            "max_slip": 0.05,
        },
    )

    right_slip_band = RewTerm(
        func=mdp.slip_in_allowed_band,
        weight=0.3,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=RIGHT_CONTACT_LINKS),
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "slip_direction",
            "min_slip": 0.0,
            "max_slip": 0.05,
        },
    )

    # Force stability (variance penalty)
    left_force_stability = RewTerm(
        func=mdp.force_variance_penalty,
        weight=-0.15,
        params={"sensor_cfg": SceneEntityCfg("left_contact_sensor")},
    )

    right_force_stability = RewTerm(
        func=mdp.force_variance_penalty,
        weight=-0.15,
        params={"sensor_cfg": SceneEntityCfg("right_contact_sensor")},
    )

    # Action smoothness
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    contact_lost = DoneTerm(
        func=mdp.all_contacts_lost,
        params={
            "left_sensor_cfg": SceneEntityCfg("left_contact_sensor"),
            "right_sensor_cfg": SceneEntityCfg("right_contact_sensor"),
            "num_steps": 15,
        },
    )

    uncontrolled_slip = DoneTerm(
        func=mdp.uncontrolled_slip_detected,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=LEFT_CONTACT_LINKS),
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "slip_direction",
            "perp_threshold": 0.1,
            "num_steps": 10,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum: start with small disturbances, increase over time."""

    pass  # TODO: Implement disturbance magnitude curriculum


@configclass
class TangentialComplianceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for tangential compliance environment."""

    scene: TangentialComplianceSceneCfg = TangentialComplianceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.lookat = (0.5, 0.0, 0.2)
