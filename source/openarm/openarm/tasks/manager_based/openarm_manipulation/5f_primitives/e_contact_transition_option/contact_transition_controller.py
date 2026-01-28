# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0

"""Contact Transition Controller - Procedural orchestration of primitives.

This is a DETERMINISTIC procedure, NOT a continuously blended policy.
It invokes primitives a/b/d with guards for safe contact transitions.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Tuple, List

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TransitionPhase(Enum):
    """Phases of a contact transition."""
    IDLE = auto()
    STABILIZE_HOLD = auto()      # Use primitive b to stabilize
    PRESHAPE_FINGER = auto()     # Use primitive d to prepare finger
    LIFT_FINGER = auto()         # Lift finger from surface
    MOVE_FINGER = auto()         # Move finger to new location
    REESTABLISH_CONTACT = auto()  # Land finger and stabilize
    VERIFY_SUCCESS = auto()       # Check transition succeeded
    FAILED = auto()


@dataclass
class TransitionGoal:
    """Goal specification for a contact transition.

    Attributes:
        finger_id: Which finger to move (0-4 for each hand)
        hand: "left" or "right"
        target_point: Target contact point in hand frame (3D)
        keep_contact_fingers: List of fingers that must maintain contact
    """
    finger_id: int
    hand: str
    target_point: torch.Tensor  # (3,) target position
    keep_contact_fingers: List[int]


class ContactTransitionController:
    """Procedural controller for contact transitions.

    This controller orchestrates trained primitives (a, b, d) to execute
    safe contact transitions. It is NOT an RL policy itself.

    Usage:
        controller = ContactTransitionController(env)
        controller.set_goal(TransitionGoal(...))
        while not controller.is_done():
            actions = controller.step()
            # Apply actions to environment
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        primitive_b_policy=None,  # Trained primitive b policy
        primitive_d_policy=None,  # Trained primitive d policy
        control_rate_hz: float = 10.0,
    ):
        """Initialize the controller.

        Args:
            env: The environment instance
            primitive_b_policy: Trained contact force hold policy
            primitive_d_policy: Trained finger synergy shape policy
            control_rate_hz: Control rate for option execution
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs

        # Store primitive policies
        # TODO: Load trained policies for primitives b and d
        self.primitive_b_policy = primitive_b_policy
        self.primitive_d_policy = primitive_d_policy

        self.control_rate_hz = control_rate_hz
        self.dt = 1.0 / control_rate_hz

        # State tracking
        self.phase = torch.full(
            (self.num_envs,), TransitionPhase.IDLE.value,
            device=self.device, dtype=torch.long
        )
        self.phase_step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.goals: List[Optional[TransitionGoal]] = [None] * self.num_envs

        # Phase timeouts (in steps)
        self.phase_timeout = {
            TransitionPhase.STABILIZE_HOLD: 50,
            TransitionPhase.PRESHAPE_FINGER: 30,
            TransitionPhase.LIFT_FINGER: 20,
            TransitionPhase.MOVE_FINGER: 40,
            TransitionPhase.REESTABLISH_CONTACT: 50,
            TransitionPhase.VERIFY_SUCCESS: 20,
        }

    def set_goal(self, env_ids: torch.Tensor, goals: List[TransitionGoal]):
        """Set transition goals for specific environments.

        Args:
            env_ids: Environment indices
            goals: List of TransitionGoal for each env in env_ids
        """
        for i, env_id in enumerate(env_ids.tolist()):
            self.goals[env_id] = goals[i]
            self.phase[env_id] = TransitionPhase.STABILIZE_HOLD.value
            self.phase_step_count[env_id] = 0

    def step(self) -> torch.Tensor:
        """Execute one step of the transition procedure.

        Returns:
            Actions for all environments (joint position targets)
        """
        # Initialize actions
        # TODO: Get action dimension from environment
        action_dim = 40  # 20 joints per hand
        actions = torch.zeros(self.num_envs, action_dim, device=self.device)

        # Process each phase
        for phase in TransitionPhase:
            if phase == TransitionPhase.IDLE or phase == TransitionPhase.FAILED:
                continue

            phase_mask = self.phase == phase.value
            if not phase_mask.any():
                continue

            phase_actions = self._execute_phase(phase, phase_mask)
            actions[phase_mask] = phase_actions

            # Increment step count
            self.phase_step_count[phase_mask] += 1

            # Check for phase transitions
            self._check_phase_transitions(phase, phase_mask)

        return actions

    def _execute_phase(
        self, phase: TransitionPhase, mask: torch.Tensor
    ) -> torch.Tensor:
        """Execute actions for a specific phase.

        Args:
            phase: Current phase
            mask: Boolean mask for environments in this phase

        Returns:
            Actions for masked environments
        """
        num_active = mask.sum().item()

        if phase == TransitionPhase.STABILIZE_HOLD:
            # Use primitive b to maintain stable hold
            if self.primitive_b_policy is not None:
                # Get observations and run policy
                # TODO: Properly interface with primitive b
                pass
            return torch.zeros(num_active, 40, device=self.device)

        elif phase == TransitionPhase.PRESHAPE_FINGER:
            # Use primitive d to prepare finger for transition
            if self.primitive_d_policy is not None:
                # TODO: Generate synergy target for lifting finger
                pass
            return torch.zeros(num_active, 40, device=self.device)

        elif phase == TransitionPhase.LIFT_FINGER:
            # Open the finger being moved
            # This is a simple scripted motion
            actions = torch.zeros(num_active, 40, device=self.device)
            # TODO: Set appropriate joint targets to lift finger
            return actions

        elif phase == TransitionPhase.MOVE_FINGER:
            # Move finger towards target
            actions = torch.zeros(num_active, 40, device=self.device)
            # TODO: Interpolate towards target position
            return actions

        elif phase == TransitionPhase.REESTABLISH_CONTACT:
            # Close finger to make contact
            # Use primitive b with relaxed grip
            if self.primitive_b_policy is not None:
                pass
            return torch.zeros(num_active, 40, device=self.device)

        elif phase == TransitionPhase.VERIFY_SUCCESS:
            # Maintain position while verifying
            return torch.zeros(num_active, 40, device=self.device)

        return torch.zeros(num_active, 40, device=self.device)

    def _check_phase_transitions(
        self, phase: TransitionPhase, mask: torch.Tensor
    ):
        """Check and execute phase transitions.

        Args:
            phase: Current phase
            mask: Boolean mask for environments in this phase
        """
        from .transition_guards import TransitionGuards

        env_ids = torch.where(mask)[0]

        # Check timeout
        timeout = self.phase_timeout.get(phase, 100)
        timed_out = self.phase_step_count[mask] >= timeout

        if phase == TransitionPhase.STABILIZE_HOLD:
            # Transition when grip is stable
            stable = TransitionGuards.is_grip_stable(self.env, env_ids)
            advance = stable | timed_out
            self._advance_phase(env_ids[advance], TransitionPhase.PRESHAPE_FINGER)

        elif phase == TransitionPhase.PRESHAPE_FINGER:
            # Transition when finger is preshaped
            ready = self.phase_step_count[mask] >= 10  # Simple condition
            advance = ready | timed_out
            self._advance_phase(env_ids[advance], TransitionPhase.LIFT_FINGER)

        elif phase == TransitionPhase.LIFT_FINGER:
            # Transition when finger is lifted
            lifted = TransitionGuards.is_finger_lifted(self.env, env_ids, finger_ids=[])
            advance = lifted | timed_out
            self._advance_phase(env_ids[advance], TransitionPhase.MOVE_FINGER)

        elif phase == TransitionPhase.MOVE_FINGER:
            # Transition when finger reaches target
            at_target = TransitionGuards.is_finger_at_target(
                self.env, env_ids, target_positions=None
            )
            advance = at_target | timed_out
            self._advance_phase(env_ids[advance], TransitionPhase.REESTABLISH_CONTACT)

        elif phase == TransitionPhase.REESTABLISH_CONTACT:
            # Transition when contact is made
            contact_made = TransitionGuards.has_finger_contact(self.env, env_ids, finger_ids=[])
            advance = contact_made | timed_out
            self._advance_phase(env_ids[advance], TransitionPhase.VERIFY_SUCCESS)

        elif phase == TransitionPhase.VERIFY_SUCCESS:
            # Check overall success
            success = TransitionGuards.is_transition_successful(self.env, env_ids)
            self.phase[env_ids[success]] = TransitionPhase.IDLE.value
            self.phase[env_ids[~success & timed_out]] = TransitionPhase.FAILED.value

    def _advance_phase(self, env_ids: torch.Tensor, next_phase: TransitionPhase):
        """Advance specified environments to next phase.

        Args:
            env_ids: Environment indices to advance
            next_phase: Next phase to enter
        """
        if len(env_ids) > 0:
            self.phase[env_ids] = next_phase.value
            self.phase_step_count[env_ids] = 0

    def is_done(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Check if transition is complete.

        Args:
            env_ids: Optional subset of environments to check

        Returns:
            Boolean tensor indicating completion
        """
        if env_ids is None:
            return (self.phase == TransitionPhase.IDLE.value) | \
                   (self.phase == TransitionPhase.FAILED.value)
        return (self.phase[env_ids] == TransitionPhase.IDLE.value) | \
               (self.phase[env_ids] == TransitionPhase.FAILED.value)

    def is_successful(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Check if transition succeeded.

        Args:
            env_ids: Optional subset of environments

        Returns:
            Boolean tensor indicating success
        """
        if env_ids is None:
            return self.phase == TransitionPhase.IDLE.value
        return self.phase[env_ids] == TransitionPhase.IDLE.value

    def reset(self, env_ids: torch.Tensor):
        """Reset controller state for specified environments.

        Args:
            env_ids: Environment indices to reset
        """
        self.phase[env_ids] = TransitionPhase.IDLE.value
        self.phase_step_count[env_ids] = 0
        for env_id in env_ids.tolist():
            self.goals[env_id] = None
