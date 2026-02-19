# 5G Lift Left v2 Rewards Guide (Current Code Base)

기준 코드:
- `source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v2/lift_left_env_cfg.py`
- `source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v2/mdp/rewards.py`
- `source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v2/config/joint_pos_env_cfg.py`
- `source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v2/mdp/actions.py`

v2는 v1 구조를 유지하면서, 접촉 판정을 `left_contact_sensor` 기반으로 확장한 버전입니다.

## 1) Phase Trigger (DexPour style)

- `λ` (approach): EE와 정적 grasp target 거리 `< 0.05`
- `μ` (grasp): `λ * (contact_fingers >= 3)`
  - v2는 기본적으로 `sensor_cfg`를 전달하여 센서 기반 contact count 사용
- `ν` (lift): `μ * (cup_z >= initial_z + 0.04)`

## 2) Contact Sensor Setup (v2 핵심)

- 센서 엔티티: `left_contact_sensor`
- 설정 위치: `.../5g_lift_left_v2/config/joint_pos_env_cfg.py`
- 현재 경로:
  - `prim_path="{ENV_REGEX_NS}/Robot/tesollo_left_.*_sensor_link"`
  - `filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup"]`
- `require_filtered_contact_matrix=True`
  - filtered matrix가 없으면 접촉 보상을 보수적으로 처리하도록 설계됨.

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
| thumb_grasp | `thumb_grasp_reward` | 15.0 | `std=0.05`, `sensor_cfg` |
| pinky_grasp | `pinky_grasp_reward` | 12.0 | `std=0.05`, `sensor_cfg` |
| synergy_grip | `synergy_grip_reward` | 20.0 | `surface-gated` |
| finger_tip_to_cup | `finger_wrap_cylinder_reward` | 0.0 | `target_radius=0.045` |
| finger_wrap_coverage | `finger_wrap_coverage_reward` | 0.0 | - |
| finger_tip_orientation | `finger_tip_orientation_reward` | 5.0 | `std=0.5` |
| lifting_object | `object_is_lifted` | 10.0 | `minimal_height=0.04`, `sensor_cfg` |
| cup_lift_progress | `cup_lift_progress_reward` | 20.0 | `std=0.05`, `sensor_cfg` |
| object_goal_tracking | `object_goal_distance` | 20.0 | `std=0.3`, `sensor_cfg` |
| object_goal_tracking_fine_grained | `object_goal_distance` | 10.0 | `std=0.1`, `sensor_cfg` |
| object_displacement | `object_displacement_penalty` | -5.0 | `threshold=0.01` |
| finger_normal_range | `finger_normal_range_penalty` | -2.0 | - |
| thumb_reaching_pose | `thumb_reaching_pose_reward` | 0.5 | `std=1.0` |
| pinky_reaching_pose | `pinky_reaching_pose_reward` | 0.5 | `std=1.0` |
| synergy_reaching_pose | `synergy_reaching_pose_reward` | 0.5 | `std=5.0` |
| thumb_tip_z | `thumb_tip_z_reward` | 10.0 | `std=0.03` |
| synergy_tip_z | `synergy_tip_z_reward` | 10.0 | `std=0.06`, `cup_height=0.09` |
| ee_descent | `ee_descent_reward` | 15.0 | `std=0.04`, `target_z_offset=0.04` |
| action_rate | `action_rate_l2` | -0.0001 | - |
| joint_vel | `joint_vel_l2` | -0.0001 | left arm+hand |

## 5) Recent Behavior-Critical Changes Reflected

- `synergy_grip_reward`:
  - `λ * surface_proximity_gate * close_reward` 구조.
  - 컵 표면 근접이 없으면 시너지 닫기 보상이 크게 줄어듦.
- 시너지 close pose spread 조인트 고정:
  - `lj_dg_2_1`, `lj_dg_3_1`, `lj_dg_4_1` = `0.0`.
- 디버그 로그 확장:
  - `touch_fingers=...`
  - `touch_links(env0): <sensor_link>:<force>N`

## 6) Termination

- `cup_dropping`: `minimum_height=-0.05`
- `cup_tipping`: `max_tilt_deg=90.0`
