# Skill Pipeline (Independent Policies)

Goal
- Train ReachIK, GraspIK, TransferIK, PouringIK separately.
- Run them sequentially in simulation to complete the full manipulation pipeline.

Overview
- Each task uses its own environment, observation space, action space, and policy checkpoint.
- A high-level orchestration script switches tasks in a fixed order and passes state (e.g., object poses) implicitly via the simulator.

Steps
1) Train each task independently
- ReachIK: train policy to move both end-effectors near cups.
- GraspIK: train policy to grasp both cups.
- TransferIK: train policy to bring cups together/align.
- PouringIK: train policy to pour.

2) Save each policy checkpoint
- Keep a clear naming scheme per task and iteration.

3) Create a sequential executor
- Pseudocode flow:
  - reset sim
  - run ReachIK policy for N steps or until success
  - switch to GraspIK policy for M steps or until success
  - switch to TransferIK policy for K steps or until success
  - switch to PouringIK policy for T steps or until success

4) Success conditions per task
- Define simple end conditions for each stage (distance threshold, grasp success, alignment, pour complete).
- If a task fails, reset or retry that stage only.

Notes
- No requirement that obs/action dims match across tasks.
- Each policy can use task-specific rewards/commands.
- Use a stable reset procedure between stages (or keep sim state if tasks are consistent).

Checklist
- [ ] Task envs configured and validated
- [ ] Checkpoints saved and named per task
- [ ] Success criteria for each task defined
- [ ] Sequential executor script built
- [ ] Stage-by-stage testing complete
