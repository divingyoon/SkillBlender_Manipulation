# 5G Lift Left v2 - 보상함수 가이드

TensorBoard에서 `5g_lift_left-v2` 리워드/종료 지표를 해석하기 위한 문서입니다.

---

## 1) Phase / Gate 전환

### reaching_progress_gate
- 기준: EE-Target 거리 기반
- 의미: reaching 단계 진행도 (0~1)
- 용도: orientation, 일부 reaching 관련 shaping 활성/비활성

### grasp_progress_gate
- 기준: 더 엄격한 EE-Target 거리 + hold steps
- 의미: grasp/lift 단계 진입 신호 (0~1)
- 용도: `grasp_contact_*`, `lifting_object`, `object_goal_tracking` 활성 조건

핵심 해석:
- `grasp_progress_gate`가 늦게 오르면 grasp/lift 보상이 거의 0으로 유지됨
- gate 전에 contact 보상이 높으면 컵을 밀어 접촉만 만드는 local optimum 가능

---

## 2) Reaching / Pre-grasp 리워드

### `reaching_object` (w=8.0)
- EE가 동적 Z target으로 접근하면 증가
- 컵 XY displacement가 커지면 내부 suppress가 걸려 감소

### `reaching_object_fine` (w=6.0)
- 고정 Z target 기준 정밀 접근
- 근접 제어 안정성 지표

### `end_effector_orientation` (w=4.0)
- reaching 구간에서 EE-Z와 컵-Z 수직 정렬 유도
- reaching 완료 후 영향 감소

### `thumb_reaching_pose` (w=1.0)
### `pinky_reaching_pose` (w=0.5)
### `synergy_reaching_pose` (w=2.0)
- grasp 전 손가락(엄지/새끼/2~4번)을 열린 기본 자세로 유지
- pre-grasp에서 손가락 꼬임/조기 클로징 방지

좋은 패턴:
- reaching 계열이 먼저 안정적으로 올라감
- pre-grasp pose 계열이 중후반까지 완만히 유지

---

## 3) Grasp/Lift 리워드

### `grasp_contact_persistence` (w=8.0)
- 최소 손가락 접촉 수 유지 보상
- `grasp_progress_gate` 이후에 의미있게 증가해야 정상

### `grasp_contact_coverage` (w=12.0)
- 접촉 손가락 coverage 보상
- 높더라도 `grasp_strict_success`/`lifting_object`가 0이면 실패 패턴

### `grasp_strict_success` (w=20.0)
- 조건: 요구 손가락 수 + 최소 리프트 높이 + hold
- 실제 grasp-lift 성공 여부를 가장 잘 반영

### `lifting_object` (w=10.0)
- 컵이 최소 높이 이상 들리면 증가

### `object_goal_tracking` (w=20.0)
### `object_goal_tracking_fine_grained` (w=10.0)
- lift 이후 목표 위치 추적 품질

좋은 패턴:
- `grasp_contact_*` 상승 이후 `grasp_strict_success`와 `lifting_object`가 따라 올라감
- 이후 `object_goal_tracking*`이 증가

---

## 4) Penalty / Termination

### `object_displacement` (w=-5.0, threshold=0.005)
- 컵 밀림 패널티
- 절대값이 큰데 strict/lift가 낮으면 “밀면서 접촉만 생성” 패턴

### `finger_normal_range` (w=-1.0)
- 비정상 관절 범위 이탈 패널티

### `action_rate` / `joint_vel` (각 w=-1e-4)
- 액션 스무딩/진동 억제(현재 v1과 동일 강도)

### 종료 조건
- `time_out`
- `cup_dropping`
- `cup_tipping` (max_tilt_deg=45)
  - 컵이 크게 기울면 즉시 에피소드 재시작

---

## 5) 실패 패턴과 권장 조정

### 패턴 A: pre-grasp에서 손가락 꼬임/조기 닫힘
- 징후: `thumb/pinky/synergy_reaching_pose`가 초반부터 낮음
- 조정 예시:
  - `env.rewards.synergy_reaching_pose.weight=2.5`
  - `env.rewards.thumb_reaching_pose.weight=1.5`

### 패턴 B: 컵을 밀기만 하고 lift 실패
- 징후: `grasp_contact_*` 높음 + `grasp_strict_success`/`lifting_object` 낮음 + `object_displacement` 음수 큼
- 조정 예시:
  - `env.rewards.object_displacement.weight=-6.0`
  - `env.rewards.grasp_contact_coverage.weight=10.0`
  - `env.grasp_soft_gate_far=0.03`

### 패턴 C: lift는 되는데 goal tracking 약함
- 징후: `lifting_object` 상승, `object_goal_tracking*` 낮음
- 조정 예시:
  - `env.rewards.object_goal_tracking.weight=24.0`
  - `env.rewards.object_goal_tracking_fine_grained.weight=12.0`

---

## 6) 권장 KPI 우선순위

1. `Episode/Episode_Reward/grasp_strict_success`
2. `Episode/Episode_Reward/lifting_object`
3. `Episode/Episode_Reward/object_displacement`
4. `Episode/Episode_Termination/time_out`, `cup_tipping`

`grasp_contact_*`는 보조 지표입니다. 이 값만 높고 strict/lift가 낮으면 성공으로 판단하지 않습니다.
