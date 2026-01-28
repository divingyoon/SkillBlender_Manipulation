# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0

"""Transition Planner - Plans regrasp sequences for contact transitions.

This planner determines which finger to move, where to move it, and
which fingers must maintain contact during the transition.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, Optional

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class TransitionPlan:
    """A planned contact transition sequence.

    Attributes:
        finger_id: Index of finger to move (0-4)
        hand: "left" or "right"
        current_contact: Current contact point in object frame
        target_contact: Target contact point in object frame
        keep_fingers: Finger indices that must maintain contact
        priority: Priority score (higher = more urgent)
    """
    finger_id: int
    hand: str
    current_contact: torch.Tensor
    target_contact: torch.Tensor
    keep_fingers: List[int]
    priority: float


class TransitionPlanner:
    """Plans contact transition sequences.

    Given current grasp state and desired grasp improvement, this planner
    determines the optimal regrasp sequence.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        min_keep_contacts: int = 3,
    ):
        """Initialize the planner.

        Args:
            env: Environment instance
            min_keep_contacts: Minimum contacts to maintain during transition
        """
        self.env = env
        self.device = env.device
        self.min_keep_contacts = min_keep_contacts

    def plan_transition(
        self,
        env_ids: torch.Tensor,
        target_contact_points: torch.Tensor,
    ) -> List[Optional[TransitionPlan]]:
        """Plan transition sequence for given environments.

        Args:
            env_ids: Environment indices
            target_contact_points: Target contact points (num_envs, num_fingers, 3)

        Returns:
            List of TransitionPlans (None if no transition needed)
        """
        plans = []

        for i, env_id in enumerate(env_ids.tolist()):
            plan = self._plan_single_transition(env_id, target_contact_points[i])
            plans.append(plan)

        return plans

    def _plan_single_transition(
        self,
        env_id: int,
        target_points: torch.Tensor,
    ) -> Optional[TransitionPlan]:
        """Plan transition for a single environment.

        Args:
            env_id: Environment index
            target_points: Target contact points (num_fingers, 3)

        Returns:
            TransitionPlan or None if no transition needed
        """
        # Get current contact state
        # TODO: Query actual contact state from environment

        # Simple heuristic: find finger with largest movement required
        # In practice, this should consider:
        # - Grasp stability impact
        # - Reachability of target
        # - Collision avoidance

        # Placeholder implementation
        finger_id = 0
        hand = "left"
        current_contact = torch.zeros(3, device=self.device)
        target_contact = target_points[finger_id] if target_points is not None else torch.zeros(3, device=self.device)
        keep_fingers = [1, 2, 3, 4]  # All other fingers

        # Check if movement is significant
        movement = torch.norm(target_contact - current_contact)
        if movement < 0.005:
            return None

        return TransitionPlan(
            finger_id=finger_id,
            hand=hand,
            current_contact=current_contact,
            target_contact=target_contact,
            keep_fingers=keep_fingers,
            priority=movement.item(),
        )

    def evaluate_transition_feasibility(
        self,
        plan: TransitionPlan,
        env_id: int,
    ) -> Tuple[bool, str]:
        """Evaluate if a planned transition is feasible.

        Args:
            plan: Transition plan to evaluate
            env_id: Environment index

        Returns:
            Tuple of (is_feasible, reason)
        """
        # Check 1: Enough remaining contacts
        if len(plan.keep_fingers) < self.min_keep_contacts:
            return False, "Insufficient remaining contacts"

        # Check 2: Target is reachable
        # TODO: Implement reachability check

        # Check 3: No collision with object during motion
        # TODO: Implement collision check

        return True, "Feasible"

    def get_intermediate_waypoints(
        self,
        plan: TransitionPlan,
        num_waypoints: int = 5,
    ) -> torch.Tensor:
        """Generate intermediate waypoints for finger motion.

        Args:
            plan: Transition plan
            num_waypoints: Number of waypoints to generate

        Returns:
            Waypoints tensor (num_waypoints, 3)
        """
        current = plan.current_contact
        target = plan.target_contact

        # Simple linear interpolation with lift
        waypoints = torch.zeros(num_waypoints, 3, device=self.device)

        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)

            # Interpolate position
            waypoints[i] = (1 - t) * current + t * target

            # Add lift in middle of trajectory (parabolic)
            lift_amount = 0.02  # 2cm lift
            lift = 4 * t * (1 - t) * lift_amount
            waypoints[i, 2] += lift  # Add to z-coordinate

        return waypoints
