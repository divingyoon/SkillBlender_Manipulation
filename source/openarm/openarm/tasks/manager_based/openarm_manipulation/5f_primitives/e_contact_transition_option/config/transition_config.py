# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0

"""Configuration parameters for Contact Transition Controller.

These are NOT training configs. Primitive E is procedural.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class TransitionControllerConfig:
    """Configuration for ContactTransitionController.

    Attributes:
        control_rate_hz: Control rate for option execution
        phase_timeouts: Timeout (in steps) for each phase
        stability_thresholds: Thresholds for stability guards
        safety_thresholds: Thresholds for safety guards
    """

    control_rate_hz: float = 10.0

    phase_timeouts: Dict[str, int] = None

    def __post_init__(self):
        if self.phase_timeouts is None:
            self.phase_timeouts = {
                "STABILIZE_HOLD": 50,
                "PRESHAPE_FINGER": 30,
                "LIFT_FINGER": 20,
                "MOVE_FINGER": 40,
                "REESTABLISH_CONTACT": 50,
                "VERIFY_SUCCESS": 20,
            }


@dataclass
class StabilityThresholds:
    """Thresholds for stability guards."""

    min_contacts: int = 3
    max_slip_velocity: float = 0.02  # m/s
    min_contact_force: float = 0.5   # N
    max_object_velocity: float = 0.05  # m/s
    max_object_angular_velocity: float = 0.1  # rad/s


@dataclass
class SafetyThresholds:
    """Thresholds for safety guards."""

    min_object_height: float = 0.0  # m (drop detection)
    max_finger_force: float = 20.0  # N
    min_remaining_contacts: int = 2


# Default configurations
DEFAULT_TRANSITION_CONFIG = TransitionControllerConfig()
DEFAULT_STABILITY_THRESHOLDS = StabilityThresholds()
DEFAULT_SAFETY_THRESHOLDS = SafetyThresholds()
