# 5G Lift Left v1 - 보상함수 가이드

TensorBoard에서 각 리워드를 해석하기 위한 문서입니다.

---

## 목차

1. [Phase 전환 메커니즘](#phase-전환-메커니즘)
2. [Reaching Phase 리워드](#reaching-phase-리워드)
3. [Grasp Phase 리워드](#grasp-phase-리워드)
4. [Lifting Phase 리워드](#lifting-phase-리워드)
5. [페널티 리워드](#페널티-리워드)
6. [리워드 요약표](#리워드-요약표)

---

## Phase 전환 메커니즘

### reaching_progress_gate
- **조건**: `EE-Target dist < 0.05m`를 **2스텝 연속** 유지
- **값**: 0 (reaching 중) → 1 (reaching 완료)
- 대부분의 grasp 리워드가 이 gate로 활성화됨

### grasp_progress_gate
- **조건**: `EE-Target dist < 0.025m`를 **4스텝 연속** 유지
- **값**: 0 (grasp 준비 중) → 1 (grasp 완료)
- lifting 관련 리워드가 이 gate로 활성화됨

---

## Reaching Phase 리워드

### 1. reaching_object
| 항목 | 값 |
|------|-----|
| **Weight** | 8.0 |
| **범위** | [0, 1] |
| **활성화** | 항상 |

**수식:**
```
target_pos = cup_pos + grasp2g_target_offset (동적 Z)
dist = ||target_pos - ee_pos||
reward = 1 - tanh(dist / 0.15)
```

**해석:**
- ↑ 증가: EE가 타겟(컵 위 grasp 위치)에 접근 중
- ↓ 감소: EE가 타겟에서 멀어짐
- ~1.0 유지: 타겟 도달 완료

---

### 2. reaching_object_fine
| 항목 | 값 |
|------|-----|
| **Weight** | 6.0 |
| **범위** | [0, 1] |
| **활성화** | 항상 |

**수식:**
```
target_pos = cup_pos + grasp2g_target_offset (고정 Z=0.08m)
dist = ||target_pos - ee_pos||
reward = 1 - tanh(dist / 0.05)
```

**해석:**
- reaching_object보다 더 민감 (std 0.05 vs 0.15)
- ↑ 증가: 타겟 근처에서 정밀하게 접근 중
- ~1.0: 매우 가까운 거리 (5cm 이내)

---

### 3. end_effector_orientation
| 항목 | 값 |
|------|-----|
| **Weight** | 4.0 |
| **범위** | [0, 1] |
| **활성화** | reaching 완료 전까지 (이후 0) |

**수식:**
```
cos_theta = dot(ee_z_axis, cup_z_axis)
error = |cos_theta|  # 0이면 수직
reward = (1 - reached_gate) × (1 - tanh(error / 0.3))
```

**해석:**
- ↑ 증가: EE Z축이 컵 Z축과 수직에 가까움 (옆에서 잡는 자세)
- ↓ 감소: EE가 컵과 평행하게 정렬됨
- reaching 완료 후 0으로 감소 (정상)

---

### 4. thumb_reaching_pose
| 항목 | 값 |
|------|-----|
| **Weight** | 1.0 |
| **범위** | [0, 1] |
| **활성화** | grasp 완료 전까지 (이후 0) |

**수식:**
```
target = {lj_dg_1_2: 1.571, lj_dg_1_3: 0, lj_dg_1_4: 0}  # 열린 자세
sq_error = Σ(pos - target)²
reward = (1 - grasp_gate) × (1 - tanh(sq_error / 1.0))
```

**해석:**
- ↑ 증가: 엄지가 열린 상태 유지 (reaching 중 좋음)
- ↓ 감소: 엄지가 닫히기 시작 또는 grasp phase 진입
- grasp 완료 후 0으로 감소 (정상)

---

### 5. pinky_reaching_pose
| 항목 | 값 |
|------|-----|
| **Weight** | 0.5 |
| **범위** | [0, 1] |
| **활성화** | grasp 완료 전까지 (이후 0) |

**수식:**
```
target = {lj_dg_5_3: 0, lj_dg_5_4: 0}  # 열린 자세
sq_error = Σ(pos - target)²
reward = (1 - grasp_gate) × (1 - tanh(sq_error / 1.0))
```

**해석:**
- thumb_reaching_pose와 동일한 패턴
- 새끼손가락이 열린 상태 유지 유도

---

## Grasp Phase 리워드

### 6. synergy_grip ⭐ (가장 높은 weight)
| 항목 | 값 |
|------|-----|
| **Weight** | 30.0 |
| **범위** | [0, 1] |
| **활성화** | 항상 (phase에 따라 다른 계산) |

**수식:**
```
grip_strength = action[-1, +1]  # -1=열림, +1=닫힘

# Reaching 중 (reached_gate=0):
open_reward = 0.02 × (1 - grip_strength) / 2

# Reaching 완료 후 (reached_gate=1):
action_reward = (grip_strength + 1) / 2
position_reward = 1 - tanh(Σ(joint_pos - close_target)² / 5.0)
close_reward = 0.5 × action_reward + 0.5 × position_reward

reward = (1 - reached_gate) × open_reward + reached_gate × close_reward
```

**해석:**
- **Reaching 중 (낮은 값 ~0.01)**: 정상, 손가락 열린 상태
- **Reaching 완료 후:**
  - ↑ 증가: 정책이 닫기 action 출력 + 실제로 닫히는 중
  - ~0.5: action은 +1인데 아직 안 닫힘 (position_reward 낮음)
  - ~1.0: action +1 + 완전히 닫힘
- **핵심 지표**: 손가락이 닫히고 있는지 확인

---

### 7. thumb_grasp
| 항목 | 값 |
|------|-----|
| **Weight** | 12.0 |
| **범위** | [0, 1] |
| **활성화** | reaching 완료 후 |

**수식:**
```
# 속도 기반 (닫히는 방향 움직임)
velocity_reward = clamp(tanh(curl_velocity / 2.0), min=0)

# 위치 기반 (닫힌 위치에 가까움)
close_target = {lj_dg_1_2: 2.5, lj_dg_1_3: -1.4, lj_dg_1_4: -1.4}
position_reward = 1 - tanh(sq_error / 2.0)

reward = reached_gate × (0.3 × velocity_reward + 0.7 × position_reward)
```

**해석:**
- ↑ 증가: 엄지가 닫히는 중 또는 닫힌 상태
- 0 유지: reaching 완료 전 (정상)
- ~1.0: 엄지 완전히 닫힘

---

### 8. pinky_grasp
| 항목 | 값 |
|------|-----|
| **Weight** | 8.0 |
| **범위** | [0, 1] |
| **활성화** | reaching 완료 후 |

**수식:**
```
close_target = {lj_dg_5_3: 1.5, lj_dg_5_4: 1.5}
# thumb_grasp와 동일한 구조 (velocity 30% + position 70%)
reward = reached_gate × (0.3 × velocity_reward + 0.7 × position_reward)
```

**해석:**
- thumb_grasp와 동일한 패턴
- 새끼손가락의 닫힘 정도 표시

---

### 9. finger_tip_to_cup (finger_wrap_cylinder)
| 항목 | 값 |
|------|-----|
| **Weight** | 12.0 |
| **범위** | [0, 1] |
| **활성화** | reaching 완료 후 |

**수식:**
```
# 각 손가락 tip의 XY 거리가 target_radius(0.04m)에 가까운지
radial_error = |tip_xy_dist - 0.04|
radial_reward = 1 - tanh(radial_error / 0.015)

# 엄지가 다른 손가락 반대편에 있는지
opposition_reward = clamp(-dot(thumb_dir, others_dir), 0, 1)

reward = reached_gate × (0.7 × radial_reward + 0.3 × opposition_reward)
```

**해석:**
- ↑ 증가: 손가락들이 컵 반경(4cm) 원형으로 배치 + 엄지 반대편
- ↓ 감소: 손가락들이 컵에서 멀거나 모여있음
- 원통형 파지 품질 지표

---

### 10. finger_wrap_coverage
| 항목 | 값 |
|------|-----|
| **Weight** | 4.0 |
| **범위** | [0, 1] |
| **활성화** | reaching 완료 후 |

**수식:**
```
# 손가락 tip들 사이의 각도 분산
pair_score = (1 - cos(angle_between_tips)) / 2
coverage_reward = average(pair_scores)

reward = reached_gate × coverage_reward
```

**해석:**
- ↑ 증가: 손가락들이 컵 주위에 고르게 분포
- ↓ 감소: 손가락들이 한쪽에 몰려있음
- ~0.5 이상이면 양호한 분포

---

### 11. finger_tip_orientation
| 항목 | 값 |
|------|-----|
| **Weight** | 5.0 |
| **범위** | [0, 1] |
| **활성화** | reaching 완료 후 |

**수식:**
```
# 각 손가락 tip의 법선이 컵 중심을 향하는지
alignment = dot(tip_normal_xy, dir_to_cup_xy)
reward = reached_gate × average(clamp(alignment, 0, 1))
```

**해석:**
- ↑ 증가: 손가락 끝이 컵 중심을 향함
- ↓ 감소: 손가락 끝이 엉뚱한 방향
- 손가락 방향 품질 지표

---

### 12. thumb_tip_z
| 항목 | 값 |
|------|-----|
| **Weight** | 10.0 |
| **범위** | [0, 1] |
| **활성화** | reaching 완료 후 |

**수식:**
```
z_error = |thumb_tip_z - cup_z|
reward = reached_gate × (1 - tanh(z_error / 0.03))
```

**해석:**
- ↑ 증가: 엄지 tip이 컵 높이에 도달
- ↓ 감소: 엄지 tip이 컵보다 높거나 낮음
- ~1.0: 엄지가 컵 높이에서 파지 준비 완료

---

### 13. synergy_tip_z
| 항목 | 값 |
|------|-----|
| **Weight** | 10.0 |
| **범위** | [0, 1] |
| **활성화** | reaching 완료 후 |

**수식:**
```
# 2번 손가락(index) tip 기준
z_error = |index_tip_z - cup_z|
reward = reached_gate × (1 - tanh(z_error / 0.03))
```

**해석:**
- ↑ 증가: 시너지 손가락들이 컵 높이에 도달
- ↓ 감소: 손가락들이 컵보다 높음
- thumb_tip_z와 함께 확인

---

## Lifting Phase 리워드

### 14. lifting_object
| 항목 | 값 |
|------|-----|
| **Weight** | 10.0 |
| **범위** | {0, 1} (binary) |
| **활성화** | grasp 완료 후 |

**수식:**
```
reward = grasp_gate × (cup_z > 0.04)
```

**해석:**
- 0: 컵이 테이블 위 또는 grasp 미완료
- 1: 컵이 4cm 이상 들어올려짐
- **핵심 성공 지표**

---

### 15. object_goal_tracking
| 항목 | 값 |
|------|-----|
| **Weight** | 20.0 |
| **범위** | [0, 1] |
| **활성화** | grasp 완료 + 컵 높이 > 4cm |

**수식:**
```
dist = ||cup_pos - goal_pos||
reward = grasp_gate × (cup_z > 0.04) × (1 - tanh(dist / 0.3))
```

**해석:**
- 0: 컵을 못 들었거나 grasp 미완료
- ↑ 증가: 컵을 들고 목표 위치로 이동 중
- ~1.0: 목표 위치 도달

---

### 16. object_goal_tracking_fine_grained
| 항목 | 값 |
|------|-----|
| **Weight** | 10.0 |
| **범위** | [0, 1] |
| **활성화** | grasp 완료 + 컵 높이 > 4cm |

**수식:**
```
reward = grasp_gate × (cup_z > 0.04) × (1 - tanh(dist / 0.1))
```

**해석:**
- object_goal_tracking보다 더 민감 (std 0.1 vs 0.3)
- 목표 근처에서 정밀 추적 유도

---

## 페널티 리워드

### 17. object_displacement
| 항목 | 값 |
|------|-----|
| **Weight** | -1.5 |
| **범위** | [0, ∞) |
| **활성화** | 항상 |

**수식:**
```
displacement = ||cup_xy - initial_cup_xy||
penalty = clamp(displacement - 0.01, min=0)
reward = -1.5 × penalty
```

**해석:**
- 0: 컵이 초기 위치에서 1cm 이내 (좋음)
- ↓ 감소 (음수): 컵이 밀려남 (나쁨)
- reaching 중 컵을 밀면 페널티

---

### 18. finger_normal_range
| 항목 | 값 |
|------|-----|
| **Weight** | -1.0 |
| **범위** | [0, ∞) |
| **활성화** | 항상 |

**수식:**
```
# 손가락 관절이 정상 범위를 벗어난 정도
violation = Σ clamp(joint_pos - limit, min=0)
reward = -1.0 × violation
```

**해석:**
- 0: 모든 관절이 정상 범위 (좋음)
- ↓ 감소 (음수): 관절이 비정상적으로 꺾임 (나쁨)

---

### 19. action_rate
| 항목 | 값 |
|------|-----|
| **Weight** | -0.0001 |
| **범위** | [0, ∞) |
| **활성화** | 항상 |

**수식:**
```
reward = -0.0001 × ||action_t - action_{t-1}||²
```

**해석:**
- ~0: 부드러운 행동 (좋음)
- ↓ 감소: 급격한 행동 변화 (약한 페널티)

---

### 20. joint_vel
| 항목 | 값 |
|------|-----|
| **Weight** | -0.0001 |
| **범위** | [0, ∞) |
| **활성화** | 항상 |

**수식:**
```
reward = -0.0001 × ||joint_velocities||²
```

**해석:**
- ~0: 느린 움직임 (좋음)
- ↓ 감소: 빠른 관절 속도 (약한 페널티)

---

## 리워드 요약표

| 리워드 | Weight | 범위 | Phase | 핵심 의미 |
|--------|--------|------|-------|----------|
| reaching_object | 8.0 | [0,1] | Reach | EE→타겟 접근 |
| reaching_object_fine | 6.0 | [0,1] | Reach | 정밀 접근 |
| end_effector_orientation | 4.0 | [0,1] | Reach | EE 자세 |
| thumb_reaching_pose | 1.0 | [0,1] | Reach | 엄지 열림 유지 |
| pinky_reaching_pose | 0.5 | [0,1] | Reach | 새끼 열림 유지 |
| **synergy_grip** | **30.0** | [0,1] | Grasp | **시너지 손가락 닫힘** |
| thumb_grasp | 12.0 | [0,1] | Grasp | 엄지 닫힘 |
| pinky_grasp | 8.0 | [0,1] | Grasp | 새끼 닫힘 |
| finger_tip_to_cup | 12.0 | [0,1] | Grasp | 원통 파지 |
| finger_wrap_coverage | 4.0 | [0,1] | Grasp | 손가락 분포 |
| finger_tip_orientation | 5.0 | [0,1] | Grasp | 손가락 방향 |
| thumb_tip_z | 10.0 | [0,1] | Grasp | 엄지 높이 |
| synergy_tip_z | 10.0 | [0,1] | Grasp | 시너지 높이 |
| lifting_object | 10.0 | {0,1} | Lift | 들어올림 성공 |
| object_goal_tracking | 20.0 | [0,1] | Lift | 목표 추적 |
| object_goal_tracking_fine | 10.0 | [0,1] | Lift | 정밀 추적 |
| object_displacement | -1.5 | ≤0 | All | 컵 밀림 페널티 |
| finger_normal_range | -1.0 | ≤0 | All | 관절 범위 페널티 |
| action_rate | -0.0001 | ≤0 | All | 행동 변화 페널티 |
| joint_vel | -0.0001 | ≤0 | All | 속도 페널티 |

---

## TensorBoard 해석 팁

### 학습 진행 체크포인트

1. **Reaching 학습 완료 신호:**
   - `reaching_object` > 0.8
   - `reaching_object_fine` > 0.6
   - `end_effector_orientation` 감소 → 0 (정상)

2. **Grasp 학습 진행 신호:**
   - `synergy_grip` 증가 (0.1 → 0.5+)
   - `thumb_grasp`, `pinky_grasp` 증가
   - `thumb_tip_z`, `synergy_tip_z` 증가

3. **Lift 학습 진행 신호:**
   - `lifting_object` > 0 (컵 들어올림 시작)
   - `object_goal_tracking` 증가

### 문제 진단

| 증상 | 가능한 원인 |
|------|------------|
| `synergy_grip` 0 유지 | reaching 미완료 또는 action 출력 문제 |
| `thumb_tip_z` 0 유지 | 엄지가 컵 높이로 안 내려감 |
| `object_displacement` 계속 감소 | 컵을 계속 밀고 있음 |
| `lifting_object` 0 유지 | grasp 실패 또는 파지력 부족 |
