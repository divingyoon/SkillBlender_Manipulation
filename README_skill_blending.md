# Skill Blending (Shared Interface)

Goal
- Implement a multi-skill policy (or skill blending system) so different skills can be blended or switched in one framework.
- Example reference: pouring3 task style.

Key Requirements
- Common observation interface (or a superset with consistent ordering).
- Common action interface (same dimension and semantics).
- Task/skill identifier injected into observations if needed.

Approach A: Unified Observation/Action Space
1) Define a shared observation spec
- Superset of all features needed by ReachIK/GraspIK/TransferIK/PouringIK.
- Fixed ordering and normalization.

2) Define a shared action spec
- Same action dimension and meaning across all tasks.
- Typically same IK action + gripper action layout.

3) Skill ID or mode flag
- Add a one-hot or scalar skill-id to the observation.
- Alternatively, separate head per skill (multi-head policy).

4) Train strategy
- Multi-task training with mixed episodes (curriculum or sampling).
- Alternatively, train each skill then distill/blend into a shared policy.

Approach B: Skill Blending Controller (Meta-policy)
1) Keep independent skill policies
- Each policy uses its own observation space (or mapped into shared format).

2) Build a blender
- Meta-policy selects or blends actions from skill policies.
- Commonly used: mixture-of-experts, gating network, or scripted switching.

3) Shared mapping layer
- Map simulator state into each skill’s required observation format.
- Map output actions into a single action interface for the simulator.

Checklist
- [ ] Shared obs/action schema finalized
- [ ] Task ID / skill ID strategy selected
- [ ] Skill blending controller implemented
- [ ] Training protocol selected (multi-task, distillation, or gating)
- [ ] Validation against pouring3 reference

Notes
- This approach requires more engineering upfront but enables smoother skill transitions.
- pouring3 is a good reference for structure, not necessarily a drop-in template.
