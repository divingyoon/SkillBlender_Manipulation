# 5G Lift Left v2 - Rewards Guide (for Orchestrator + Ollama)

이 문서는 `5g_lift_left-v2` 보상/게이트 구조를 빠르게 점검하고,
`atuo/orchestrator.py`가 override를 자동 생성할 때 참고하도록 만든 가이드입니다.

## 1) 실행/재개 체크

아래 명령은 현재 코드 기준으로 유효합니다.

```bash
python3 /home/user/rl_ws/SkillBlender_Manipulation/atuo/orchestrator.py \
  --config /home/user/rl_ws/SkillBlender_Manipulation/atuo/config/experiment.json \
  --agent rl_games_cfg_entry_point \
  --task 5g_lift_left-v2 \
  --resume_from test1 \
  --resume_checkpoint pipeline_left_5g_lift_left_v1.pth \
  --num_envs=256 \
  --gui
```

핵심 동작:
- `agent`가 `rl_games_`로 시작하면 train script/log root를 자동으로 `rl_games` 경로로 전환.
- `resume_from`는 run name(`testN`) 또는 절대경로 모두 허용.
- `resume_checkpoint`는 `log_root` 하위에서 파일명을 검색해 task/prefix 우선으로 선택.

## 2) Phase 게이트 (중요)

v2 게이트는 `reaching -> grasp` 순서로 동작합니다.

- `reaching_progress_gate`
- `grasp_progress_gate`

이번 패치에서 `grasp_progress_gate`에 아래 안전장치를 추가했습니다.
- orientation gate: EE-컵 정렬 품질이 낮으면 grasp gate 억제
- displacement safety gate: 컵 XY 밀림이 크면 grasp gate 억제
- soft gate 완화폭 축소: 거리만으로 조기 grasp 활성화되는 문제 완화

## 3) 현재 주요 보상 항목 (v2)

### Reaching/Pre-grasp
- `reaching_object` (w=8.0)
- `reaching_object_fine` (w=6.0)
- `end_effector_orientation` (w=16.0, 증가)
- `thumb_reaching_pose` (w=1.0)
- `pinky_reaching_pose` (w=0.5)
- `synergy_reaching_pose` (w=2.0)
- `pregrasp_contact_penalty` (w=-6.0, 신규)

### Grasp/Lift
- `grasp_contact_persistence` (w=6.0)
- `grasp_contact_coverage` (w=8.0)
- `grasp_strict_success` (w=18.0)
- `lifting_object` (w=10.0)

### Goal/Regularization
- `object_goal_tracking` (w=20.0)
- `object_goal_tracking_fine_grained` (w=10.0)
- `object_displacement` (w=-5.0)
- `finger_normal_range` (w=-1.0)
- `action_rate` (w=-1e-4, v1과 동일)
- `joint_vel` (w=-1e-4, v1과 동일)

## 4) 컨택 센서 해석 (테이블 혼동 방지)

v2는 `left_contact_sensor`에 다음 필터를 사용합니다.
- `filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup"]`

즉, 정상 상태에서는 컵 이외(테이블/자기충돌) 접촉이 grasp contact reward로 들어오지 않습니다.
또한 이번 패치로 `force_matrix_w`(필터된 접촉행렬)가 없으면 contact reward를 0 처리할 수 있도록 했습니다.

## 5) Termination

- `cup_tipping`: `max_tilt_deg=35.0` (강화)
- `cup_dropping`: 높이 하한 미만이면 종료

목표: 컵이 쓰러지는 정책은 즉시 episode 재시작.

## 6) Ollama가 조정할 우선 파라미터

### 1순위: 게이트
- `env.grasp_soft_prefactor`
- `env.grasp_orientation_gate_min_reward`
- `env.grasp_orientation_gate_full_reward`
- `env.grasp_displacement_free_threshold`
- `env.grasp_displacement_suppress_scale`
- `env.grasp_switch_threshold`
- `env.grasp_switch_hold_steps`

### 2순위: 정렬/접근
- `env.rewards.end_effector_orientation.weight`
- `env.reach_soft_gate_near`
- `env.reach_soft_gate_far`
- `env.reach_switch_threshold`
- `env.reach_switch_hold_steps`

### 3순위: grasp 보상 균형
- `env.rewards.grasp_contact_persistence.weight`
- `env.rewards.grasp_contact_coverage.weight`
- `env.rewards.grasp_strict_success.weight`
- `env.rewards.pregrasp_contact_penalty.weight`

### 4순위: 안전/종료
- `env.terminations.cup_tipping.params.max_tilt_deg`

## 7) 권장 override 예시

```json
[
  "env.grasp_soft_prefactor=0.15",
  "env.grasp_orientation_gate_min_reward=0.30",
  "env.grasp_orientation_gate_full_reward=0.80",
  "env.grasp_displacement_free_threshold=0.008",
  "env.rewards.end_effector_orientation.weight=20.0",
  "env.rewards.pregrasp_contact_penalty.weight=-8.0"
]
```

## 8) 관찰 포인트 (정성평가 대응)

- 엄지 제외 다른 손가락이 테이블 근처에 머물며 보상 획득하는지
  - `grasp_contact_*` 상승 vs `pregrasp_contact_penalty` 동시 상승 여부 확인
- 컵을 밀며 잡는지
  - `object_displacement` 악화와 `grasp_progress_gate` 상승 타이밍 동시 발생 여부 확인
- 정렬 후 접근하는지
  - `end_effector_orientation`이 먼저 상승하고 이후 contact 계열이 따라오는지 확인
