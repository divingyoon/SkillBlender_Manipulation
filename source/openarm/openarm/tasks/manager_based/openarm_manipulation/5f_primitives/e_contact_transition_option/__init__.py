# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0

"""Primitive E: Contact Transition Option - Procedural contact set transitions.

IMPORTANT: This is NOT a continuous RL policy. It is a deterministic procedural
skill (option/event) that orchestrates primitives a/b/d with guards to perform
safe contact transitions (regrasp maneuvers).

Execution flow:
1. Monitor contact state and trigger conditions
2. Plan transition: identify which finger to move and where
3. Execute sequence:
   a. Use primitive d to pre-shape the moving finger
   b. Use primitive b to stabilize remaining contacts
   c. Lift finger (small arm motion via primitive a hints)
   d. Move finger to new contact point
   e. Re-establish contact using primitive b
4. Verify success via guards

This module provides:
- ContactTransitionController: Main procedural controller
- TransitionGuards: Safety checks and success conditions
- TransitionPlanner: Plans regrasp sequences

NOT trainable via RL. The primitives (a, b, d) are trained separately.
"""

from .contact_transition_controller import ContactTransitionController
from .transition_guards import TransitionGuards
from .transition_planner import TransitionPlanner
