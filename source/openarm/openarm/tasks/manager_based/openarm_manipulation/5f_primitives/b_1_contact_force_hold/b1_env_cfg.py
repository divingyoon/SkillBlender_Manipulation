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

"""B1 wrapper env cfg: lift-start + time-varying payload, reusing Primitive B specs."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from openarm.tasks.manager_based.openarm_manipulation.primitives.b_contact_force_hold.contact_force_hold_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from openarm.tasks.manager_based.openarm_manipulation.primitives.b_contact_force_hold.config.joint_pos_env_cfg import (
    OpenArmContactForceHoldEnvCfg,
)

from .mdp import events as b1_mdp


@configclass
class B1EventCfg(EventCfg):
    """Event extensions for B1: lift-start + payload disturbance."""

    # Time-varying payload disturbance (downward force).
    payload_disturbance = EventTerm(
        func=b1_mdp.apply_payload_disturbance,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "object_cfg": SceneEntityCfg("object"),
            "delta_mass_range": (0.0, 0.4),
            "ramp_delay_steps": 0,
            "ramp_steps": 2000,
            "period_steps": 200,
            "phase_offset": 0,
        },
    )

    # payload_mass_ramp = EventTerm(
    #     func=b1_mdp.apply_payload_mass_ramp,
    #     mode="interval",
    #     interval_range_s=(0.02, 0.02),
    #     params={
    #         "object_cfg": SceneEntityCfg("object", body_names=".*"),
    #         "start_mass": 0.2,
    #         "end_mass": 1.0,
    #         "delay_steps": 50,
    #         "ramp_steps": 100,
    #         "recompute_inertia": True,
    #     },
    # )

    def __post_init__(self):
        super().__post_init__()
        # Override reset term to inject lift-start branch.
        self.reset_contact_state = self.reset_contact_state.replace(
            func=b1_mdp.reset_contact_force_hold_state_b1,
            params={
                **self.reset_contact_state.params,
                "lift_prob_init": 0.0,
                "lift_prob_final": 0.0,
                "lift_prob_steps": 0,
                "lift_offset_b1": {"x": (0.09, 0.09), "y": (0.005, 0.005), "z": (0.12, 0.12)},
                "lift_offset_b2": {"x": (0.10, 0.10), "y": (0.005, 0.005), "z": (0.10, 0.10)},
                "base_lift_z": 0.0,
            },
        )


@configclass
class OpenArmContactForceHoldB1EnvCfg(OpenArmContactForceHoldEnvCfg):
    """B1 environment: same obs/act/reward/termination as B, only events differ."""

    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    events: B1EventCfg = B1EventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Remove the table for B1 while keeping reset identical to B.
        self.scene.table = None
        self.scene.num_envs = 4096
        # Keep reset mass fixed; per-episode ramp is handled by payload_mass_ramp.
        self.events.object_mass.params["mass_distribution_params"] = (0.2, 0.2)
        self.curriculum.object_mass_range = None


@configclass
class OpenArmContactForceHoldB1EnvCfg_PLAY(OpenArmContactForceHoldB1EnvCfg):
    """Play/evaluation configuration for B1."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
