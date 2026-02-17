# 5G Lift Left v1 Rewards Guide (Code-Aligned)

이 문서는 아래 코드 기준으로 작성되었습니다.
- `source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v1/lift_left_env_cfg.py`
- `source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v1/mdp/rewards.py`

기존 문서와 달리, 현재 구현의 핵심 게이트는 대부분 `reaching_progress_gate`가 아니라 **DexPour 스타일 트리거**(`λ`, `μ`, `ν`)입니다.

## 1) 트리거(Phase Gate) 정의

### λ (approach trigger)
- 정의: `EE-정적 grasp target 거리 < 0.05m`
- 구현: `_approach_trigger(..., d_approach=0.05)`
- 특징: hold-step 없음, 이진(0/1)

### μ (grasp trigger)
- 정의: `λ * (접촉 fingertip 수 >= 4)`
- 구현: `_grasp_trigger(..., min_contacts=4, cup_radius=0.05, contact_threshold=0.02)`
- 접촉 판정:
  - XY에서 컵 표면 반경(0.05m) 대비 거리 오차 < 0.02
  - fingertip Z가 `[cup_z, cup_z + 0.10]` 범위

### ν (lift trigger)
- 정의: `μ * (cup_z >= episode_initial_z + 0.04)`
- 구현: `_lift_trigger(..., h_lift=0.04)`

참고: `_reaching_progress_gate`, `_grasp_progress_gate`도 존재하지만, 현재 reward 중 실질적으로 사용되는 곳은 `object_displacement_penalty`(가중 혼합)입니다.

## 2) 타겟 포인트(Grasp Target) 정의

### 정적 grasp target
- `grasp2g_target_offset = (0.01, -0.06, 0.08)`를 컵 로컬 기준으로 월드 변환해 사용
- `object_ee_distance_fine`, λ 판정 등에 사용

### 동적 Z grasp target (reaching_object 전용)
- XY 정렬이 안 된 초기에는 높은 Z에서 접근, XY 정렬될수록 Z를 0.08로 하강
- 기본 파라미터:
  - `reach_dynamic_z_high = 0.25`
  - `reach_dynamic_xy_hi = 0.10`
  - `reach_dynamic_xy_lo = 0.03`
  - `reach_dynamic_xy_gate = 0.03`
  - `reach_dynamic_z_descent_rate = 0.001`
- 추가 억제: 접근 중 컵 XY 밀림이 크면 `reaching_object`가 `exp(-excess/scale)`로 감쇠

## 3) 보상 항목 전체 (현재 22개)

아래 표의 `함수`/`weight`가 실제 학습에 적용됩니다.

| Term | 함수 | Weight | 활성 게이트 |
|---|---|---:|---|
| reaching_object | `object_ee_distance(std=0.15)` | 8.0 | 항상 |
| reaching_object_fine | `object_ee_distance_fine(std=0.065)` | 10.0 | 항상 |
| end_effector_orientation | `eef_z_perpendicular_object_z(std=0.3)` | 4.0 | 항상 |
| thumb_grasp | `thumb_grasp_reward(std=2.0)` | 8.0 | `λ=1` |
| pinky_grasp | `pinky_grasp_reward(std=2.0)` | 5.0 | `λ=1` |
| synergy_grip | `synergy_grip_reward(action_name=left_hand_action)` | 14.0 | `λ=1` |
| finger_tip_to_cup | `finger_wrap_cylinder_reward(...)` | 14.0 | `λ=1` |
| finger_wrap_coverage | `finger_wrap_coverage_reward()` | 4.0 | `λ=1` |
| finger_tip_orientation | `finger_tip_orientation_reward(std=0.5)` | 5.0 | `λ=1` |
| lifting_object | `object_is_lifted(minimal_height=0.04)` | 10.0 | `μ` 내부 사용 |
| object_goal_tracking | `object_goal_distance(std=0.3, minimal_height=0.04)` | 20.0 | `ν=1` |
| object_goal_tracking_fine_grained | `object_goal_distance(std=0.1, minimal_height=0.04)` | 10.0 | `ν=1` |
| object_displacement | `object_displacement_penalty(threshold=0.01)` | -4.0 | 항상 (단, grasp 진행도 혼합) |
| finger_normal_range | `finger_normal_range_penalty()` | -1.0 | 항상 |
| thumb_reaching_pose | `thumb_reaching_pose_reward(std=1.0)` | 0.5 | `λ=0` |
| pinky_reaching_pose | `pinky_reaching_pose_reward(std=1.0)` | 0.5 | `λ=0` |
| synergy_reaching_pose | `synergy_reaching_pose_reward(std=5.0)` | 0.5 | `λ=0` |
| thumb_tip_z | `thumb_tip_z_reward(std=0.06, cup_height=0.08)` | 10.0 | `λ=1` |
| synergy_tip_z | `synergy_tip_z_reward(std=0.06, cup_height=0.08)` | 10.0 | `λ=1` |
| ee_descent | `ee_descent_reward(std=0.04, target_z_offset=0.04)` | 15.0 | `λ=1` |
| action_rate | `action_rate_l2` | -0.0001 | 항상 |
| joint_vel | `joint_vel_l2` | -0.0001 | 항상 |

## 4) 항목별 해석 포인트

### Reaching 계열
- `reaching_object`: 동적 Z 타겟에 대한 거리 보상 `1 - tanh(dist/0.15)`.
- `reaching_object_fine`: 정적 grasp 타겟에 대한 정밀 거리 보상 `1 - tanh(dist/0.065)`.
- `end_effector_orientation`: EE z축과 컵 z축이 수직(직교)에 가까울수록 증가.

### Reaching pose 유지(λ=0)
- `thumb_reaching_pose`, `pinky_reaching_pose`, `synergy_reaching_pose`:
  - 손가락을 열린 목표 자세(주로 0 또는 지정된 open target) 근처로 유지.
  - `λ`가 1이 되면 자동으로 0으로 꺼짐.

### Grasp 형성(λ=1)
- `synergy_grip`:
  - `left_hand_action`의 raw scalar(`-1=open, +1=close`)와 실제 관절 닫힘 자세를 50:50 결합.
- `thumb_grasp`:
  - 엄지 닫힘 속도(양의 방향만) + 닫힘 자세 + 엄지 opposition(thumb vs other fingers 반대방향) + 접촉 게이트(floor 0.2).
- `pinky_grasp`:
  - 새끼 닫힘 속도/자세 결합 + 접촉 게이트(floor 0.2).
- `finger_tip_to_cup`(실제 함수는 `finger_wrap_cylinder_reward`):
  - 컵 반경(target_radius=0.04) 링 근접 + thumb opposition 결합.
- `finger_wrap_coverage`:
  - fingertip 각도 분산(쏠림 방지) 점수.
- `finger_tip_orientation`:
  - fingertip normal이 컵 중심을 향할수록 증가.
- `thumb_tip_z`, `synergy_tip_z`:
  - 각각 엄지/검지 tip Z를 컵 상단 높이(`cup_z + 0.08`)로 유도, XY 멀면 감쇠.
- `ee_descent`:
  - EE z를 `cup_z + 0.04`로 추가 하강 유도, XY 멀면 감쇠.

### Lift/Goal
- `lifting_object`:
  - `μ`를 만족한 상태에서 컵이 초기 높이 대비 0.04m 이상 올라가면 1.
- `object_goal_tracking`, `object_goal_tracking_fine_grained`:
  - `ν=1`일 때만 목표 pose 거리 보상 활성화.

### Penalty
- `object_displacement`:
  - `excess = max(||cup_xy - init_xy|| - 0.01, 0)`
  - `penalty = (excess / displacement_penalty_scale)^power`
  - 기본 파라미터: `scale=0.02`, `power=2.0`
  - grasp 진행 게이트와 혼합(`displacement_penalty_gate_mix=0.5`)되어 후반으로 갈수록 강해짐.
- `finger_normal_range`:
  - 지정된 thumb/pinky 정상 범위를 벗어난 라디안 위반량 합.
- `action_rate`, `joint_vel`:
  - 약한 L2 regularization.

## 5) TensorBoard 해석 체크리스트

### 정상 진행 패턴
1. 초반:
- `reaching_object`, `reaching_object_fine` 상승
- `thumb_reaching_pose`, `pinky_reaching_pose`, `synergy_reaching_pose` 유지

2. λ 전환 후:
- 위 3개 reaching_pose 계열 감소(비활성)
- `synergy_grip`, `thumb_grasp`, `pinky_grasp`, `ee_descent`, `thumb_tip_z`, `synergy_tip_z` 상승

3. grasp 성립 후:
- `lifting_object`가 간헐적으로 1에 도달
- 안정 lift가 되면 `object_goal_tracking` 계열 증가

### 자주 보이는 실패 패턴
- `reaching_object`는 높지만 `thumb_grasp/pinky_grasp`가 정체:
  - λ는 켜졌지만 실제 접촉/opposition 부족 가능성 큼.
- `lifting_object`가 0 근처 고정:
  - μ 조건(접촉 수>=4) 또는 lift 높이 0.04m 미달.
- `object_displacement`가 크게 증가:
  - 컵을 밀고 있으며, 후반으로 갈수록 패널티 영향 커짐.

## 6) 기존 문서 대비 핵심 수정 사항

- `reaching_object_fine` weight: `6.0 -> 10.0`
- `synergy_grip` weight: `30.0 -> 14.0`
- `thumb_grasp` weight: `12.0 -> 8.0`
- `pinky_grasp` weight: `8.0 -> 5.0`
- `finger_tip_to_cup` weight: `12.0 -> 14.0`
- `object_displacement` weight: `-1.5 -> -4.0`
- 누락되어 있던 `synergy_reaching_pose`, `ee_descent` 추가
- `grasp_progress_gate` 중심 설명을 제거하고, 실제 적용되는 `λ/μ/ν` 기반으로 정리
