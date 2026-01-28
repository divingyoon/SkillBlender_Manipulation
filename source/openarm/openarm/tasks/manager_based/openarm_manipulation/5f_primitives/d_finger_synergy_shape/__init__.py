# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0

"""Primitive D: Finger Joint Target Tracking - Hand configuration via joint space.

Joints scope: hand only (20 DOF per hand)
Action space: joint position targets
Control rate: 50 Hz

Observations:
- State: hand joint positions/velocities, self-collision proxy, joint limit margins
- Goal: target joint positions (20 DOF per hand)

Rewards:
- Joint target tracking
- Self-collision penalty
- Joint limit penalty
- Action smoothness

Terminations:
- Success: shape within tolerance for N steps
- Failure: severe self-collision
- Timeout: 6 seconds

Curriculum:
- Stage 1: No object (free space shaping)
- Stage 2: With object in hand
- Stage 3: With disturbances
"""

from .finger_synergy_shape_env_cfg import FingerSynergyShapeEnvCfg

# Ensure gym registration happens when this package is imported.
from . import config  # noqa: F401
