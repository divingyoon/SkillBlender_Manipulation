# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0

"""Transition Guards - Safety checks and success conditions for contact transitions.

These guards determine when to transition between phases and verify success.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TransitionGuards:
    """Static guard functions for contact transition safety and success."""

    @staticmethod
    def is_grip_stable(
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        min_contacts: int = 3,
        max_slip: float = 0.02,
        min_force: float = 0.5,
    ) -> torch.Tensor:
        """Check if grip is stable enough to begin transition.

        Args:
            env: Environment instance
            env_ids: Environment indices to check
            min_contacts: Minimum number of required contacts
            max_slip: Maximum allowed slip velocity
            min_force: Minimum contact force per finger

        Returns:
            Boolean tensor indicating stable grip
        """
        # TODO: Implement using contact sensor data
        # For now, return placeholder
        return torch.ones(len(env_ids), dtype=torch.bool, device=env.device)

    @staticmethod
    def is_finger_lifted(
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        finger_ids: List[int],
        min_clearance: float = 0.01,
    ) -> torch.Tensor:
        """Check if specified finger has lifted from object.

        Args:
            env: Environment instance
            env_ids: Environment indices
            finger_ids: Fingers to check
            min_clearance: Minimum distance from object surface

        Returns:
            Boolean tensor indicating finger is lifted
        """
        # TODO: Implement by checking fingertip-object distance
        return torch.ones(len(env_ids), dtype=torch.bool, device=env.device)

    @staticmethod
    def is_finger_at_target(
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        target_positions: Optional[torch.Tensor],
        tolerance: float = 0.01,
    ) -> torch.Tensor:
        """Check if finger has reached target position.

        Args:
            env: Environment instance
            env_ids: Environment indices
            target_positions: Target positions (num_envs, 3)
            tolerance: Position tolerance

        Returns:
            Boolean tensor indicating target reached
        """
        # TODO: Implement by checking fingertip position
        return torch.ones(len(env_ids), dtype=torch.bool, device=env.device)

    @staticmethod
    def has_finger_contact(
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        finger_ids: List[int],
        min_force: float = 0.1,
    ) -> torch.Tensor:
        """Check if specified finger has made contact.

        Args:
            env: Environment instance
            env_ids: Environment indices
            finger_ids: Fingers to check
            min_force: Minimum contact force threshold

        Returns:
            Boolean tensor indicating contact made
        """
        # TODO: Implement using contact sensor data
        return torch.ones(len(env_ids), dtype=torch.bool, device=env.device)

    @staticmethod
    def is_object_stable(
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        max_velocity: float = 0.05,
        max_angular_velocity: float = 0.1,
    ) -> torch.Tensor:
        """Check if object is stable (not dropped, not moving excessively).

        Args:
            env: Environment instance
            env_ids: Environment indices
            max_velocity: Maximum allowed linear velocity
            max_angular_velocity: Maximum allowed angular velocity

        Returns:
            Boolean tensor indicating object stability
        """
        # TODO: Implement using object velocity data
        return torch.ones(len(env_ids), dtype=torch.bool, device=env.device)

    @staticmethod
    def is_transition_successful(
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Check if overall transition was successful.

        Success criteria:
        - Object not dropped
        - Required contacts maintained
        - New contact established
        - Grip is stable

        Args:
            env: Environment instance
            env_ids: Environment indices

        Returns:
            Boolean tensor indicating success
        """
        object_stable = TransitionGuards.is_object_stable(env, env_ids)
        grip_stable = TransitionGuards.is_grip_stable(env, env_ids)

        return object_stable & grip_stable

    @staticmethod
    def should_abort_transition(
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Check if transition should be aborted for safety.

        Abort conditions:
        - Object dropped (height below threshold)
        - Excessive object motion
        - All contacts lost

        Args:
            env: Environment instance
            env_ids: Environment indices

        Returns:
            Boolean tensor indicating abort condition
        """
        # TODO: Implement safety checks
        return torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
