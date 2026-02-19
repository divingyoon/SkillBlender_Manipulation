# 5G Lift Left v1 Rewards Guide (Current Code Base)

기준 코드:
- `source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v1/lift_left_env_cfg.py`
- `source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v1/mdp/rewards.py`
- `source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v1/mdp/actions.py`

## 1) Phase Trigger (DexPour style)

- `λ` (approach): EE와 정적 grasp target 거리 `< 0.05`
  - `_approach_trigger(..., d_approach=0.05)`
- `μ` (grasp): `λ * (contact_fingers >= 3)`
  - `_grasp_trigger(..., min_contacts=3, cup_radius=0.045, contact_threshold=0.02)`
- `ν` (lift): `μ * (cup_z >= initial_z + 0.04)`
  - `_lift_trigger(..., h_lift=0.04)`

## 2) Grasp Target / Reach Params

- `grasp2g_target_offset = (0.01, -0.06, 0.08)`
- 동적 접근 Z 파라미터:
  - `reach_dynamic_z_high=0.25`
  - `reach_dynamic_xy_hi=0.10`
  - `reach_dynamic_xy_lo=0.03`
  - `reach_dynamic_xy_gate=0.03`
  - `reach_dynamic_z_descent_rate=0.001`
- 접근 중 컵 밀림 억제:
  - `reach_displacement_free_threshold=0.015`
  - `reach_displacement_suppress_scale=0.03`

## 3) Stage-by-Stage (중요)

- Stage A: `λ=0` (Approach 전/중)
  - 핵심: `reaching_object`, `reaching_object_fine`, `end_effector_orientation`
  - pose 유지: `thumb_reaching_pose`, `pinky_reaching_pose`, `synergy_reaching_pose`
- Stage B: `λ=1, μ=0` (Approach 완료, grasp 미성립)
  - 핵심: `thumb_grasp`, `pinky_grasp`, `synergy_grip`, `thumb_tip_z`, `synergy_tip_z`, `ee_descent`
  - 즉, `ee_descent`는 이 단계부터 이미 활성됨 (`λ` 게이트)
- Stage C: `μ=1, ν=0` (grasp 성립, 아직 lift 미성립)
  - 핵심: `lifting_object`, `cup_lift_progress`
- Stage D: `ν=1` (lift 성립)
  - 핵심: `object_goal_tracking`, `object_goal_tracking_fine_grained`
- 전 단계 공통 패널티/정규화:
  - `object_displacement`, `finger_normal_range`, `action_rate`, `joint_vel`

## 4) Reward Terms (Current)

| Term | Function | Weight | Key Params |
|---|---|---:|---|
| reaching_object | `object_ee_distance` | 8.0 | `std=0.15` |
| reaching_object_fine | `object_ee_distance_fine` | 10.0 | `std=0.065` |
| end_effector_orientation | `eef_z_perpendicular_object_z` | 4.0 | `std=0.3` |
| thumb_grasp | `thumb_grasp_reward` | 15.0 | `std=0.05` |
| pinky_grasp | `pinky_grasp_reward` | 12.0 | `std=0.05` |
| synergy_grip | `synergy_grip_reward` | 20.0 | `surface-gated` |
| finger_tip_to_cup | `finger_wrap_cylinder_reward` | 0.0 | `target_radius=0.045` |
| finger_wrap_coverage | `finger_wrap_coverage_reward` | 0.0 | - |
| finger_tip_orientation | `finger_tip_orientation_reward` | 5.0 | `std=0.5` |
| lifting_object | `object_is_lifted` | 10.0 | `minimal_height=0.04` |
| cup_lift_progress | `cup_lift_progress_reward` | 20.0 | `std=0.05` |
| object_goal_tracking | `object_goal_distance` | 20.0 | `std=0.3, minimal_height=0.04` |
| object_goal_tracking_fine_grained | `object_goal_distance` | 10.0 | `std=0.1, minimal_height=0.04` |
| object_displacement | `object_displacement_penalty` | -5.0 | `threshold=0.01` |
| finger_normal_range | `finger_normal_range_penalty` | -2.0 | - |
| thumb_reaching_pose | `thumb_reaching_pose_reward` | 0.5 | `std=1.0` |
| pinky_reaching_pose | `pinky_reaching_pose_reward` | 0.5 | `std=1.0` |
| synergy_reaching_pose | `synergy_reaching_pose_reward` | 0.5 | `std=5.0` |
| thumb_tip_z | `thumb_tip_z_reward` | 8.0 | `std=0.10` |
| synergy_tip_z | `synergy_tip_z_reward` | 8.0 | `std=0.06, cup_height=0.09` |
| ee_descent | `ee_descent_reward` | 10.0 | `std=0.04, target_z_offset=0.04` |
| action_rate | `action_rate_l2` | -0.0001 | - |
| joint_vel | `joint_vel_l2` | -0.0001 | left arm+hand |

## 5) Recent Behavior-Critical Changes Reflected

- `synergy_grip_reward`는 이제 단순 action-only가 아니라,
  - `λ` + 시너지 fingertip(2/3/4)의 컵 표면 근접 gate를 곱함.
  - 즉, 허공에서 닫기만 해서는 높은 보상을 받기 어려움.
- 시너지 close pose에서 spread 조인트(`lj_dg_2_1`, `lj_dg_3_1`, `lj_dg_4_1`)는 `0.0`으로 고정.

## 6) Termination

- `cup_dropping`: `minimum_height=-0.05`
- `cup_tipping`: `max_tilt_deg=90.0`
