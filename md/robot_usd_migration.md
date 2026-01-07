# 새로운 로봇(USD) 교체 가이드

이 문서는 Isaac Lab 환경에서 기존에 사용하던 로봇을 새로운 USD 파일의 로봇으로 교체하는 과정을 설명합니다.

`openarm_bimanual` 로봇을 `openarm_tesollo_t1.usd` 로봇으로 교체하는 예시를 통해 설명합니다.

## 과정 요약

로봇을 교체하기 위해서는 단순히 USD 파일 경로만 바꾸는 것이 아니라, 로봇의 상세 설정(관절, 액추에이터, 엔드 이펙터 등)을 모두 새로운 로봇에 맞게 변경해야 합니다.

주요 단계는 다음과 같습니다.

1.  **로봇 정보 확인**: 교체할 로봇의 URDF 또는 USD 파일을 분석하여 제어에 필요한 '조인트(Joint)' 이름과 추적에 필요한 '바디/링크(Body/Link)' 이름을 정확히 파악합니다.
2.  **환경 설정 파일 수정**: 로봇을 스폰하고 제어하는 환경 설정 스크립트(`joint_pos_env_cfg.py`)를 수정합니다.

---

## 1단계: 새로운 로봇 정보 확인

새로운 로봇을 제어하고 상태를 추적하기 위해 다음 두 가지 핵심 정보를 알아야 합니다.

-   **조인트(Joint) 이름**: 액추에이터가 제어할 관절들의 이름입니다. `openarm_tesollo.urdf` 파일을 분석하여 다음과 같은 제어 가능한 조인트 이름들을 확인했습니다.
    -   **팔 조인트**: `openarm_left_joint[1-7]`, `openarm_right_joint[1-7]`
    -   **그리퍼 조인트**: `lj_dg_...`, `rj_dg_...`

-   **바디/링크(Body/Link) 이름**: 엔드 이펙터(End-Effector)처럼 위치를 추적해야 하는 로봇 몸체의 이름입니다. `openarm_tesollo_t1.usd` 파일에는 다음과 같이 엔드 이펙터가 설정되어 있음을 확인했습니다.
    -   **왼쪽 엔드 이펙터**: `ll_dg_ee`
    -   **오른쪽 엔드 이펙터**: `rl_dg_ee`

## 2단계: 환경 설정 파일 수정

위에서 확인한 정보들을 바탕으로, 로봇을 불러오는 주 설정 파일인 `joint_pos_env_cfg.py`을 수정합니다.

-   **대상 파일**: `/home/user/rl_ws/openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/approach/config/joint_pos_env_cfg.py`

-   **수정 내용**: `OpenArmApproachEnvCfg` 클래스의 `__post_init__` 메서드 전체를 새로운 로봇에 맞게 교체합니다.

### 변경된 `__post_init__` 메서드 상세 설명

#### 1. 로봇 정의 (`self.scene.robot`)

기존 `OPEN_ARM_HIGH_PD_CFG` 설정을 사용하는 대신, `ArticulationCfg`를 사용하여 `openarm_tesollo_t1.usd` 로봇을 직접 정의합니다.

```python
# ...
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
# ...

# self.scene.robot 설정 부분
self.scene.robot = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    # 1. 새로운 USD 파일 경로로 변경
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/user/rl_ws/openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/usds/openarm_bimanual/openarm_tesollo_t1.usd",
        # ... 기타 물리 속성
    ),
    # 2. 새로운 로봇의 조인트 이름에 맞게 초기 상태 설정
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "openarm_left_joint1": 0.0,
            # ... (팔 조인트는 이름이 동일)
            "openarm_right_joint7": 0.0,
            "lj_dg_.*": 0.0,  # 그리퍼 조인트 이름을 새 이름(lj_dg_...)으로 변경
            "rj_dg_.*": 0.0,  # 그리퍼 조인트 이름을 새 이름(rj_dg_...)으로 변경
        },
    ),
    # 3. 새로운 조인트 이름에 맞게 액추에이터 설정
    actuators={
        "openarm_arm": ImplicitActuatorCfg(
            joint_names_expr=["openarm_left_joint[1-7]", "openarm_right_joint[1-7]"],
            # ...
        ),
        "openarm_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "lj_dg_.*",  # 그리퍼 조인트 actuator가 새 조인트 이름을 바라보도록 변경
                "rj_dg_.*",
            ],
            # ...
        ),
    },
    # ...
)
```

#### 2. 보상 및 커맨드 설정 (`rewards`, `commands`)

보상 함수와 커맨드 생성기가 엔드 이펙터의 위치를 올바르게 추적할 수 있도록 **바디 이름**을 `ll_dg_ee`와 `rl_dg_ee`로 변경합니다.

```python
# 보상(rewards) 설정에서 엔드 이펙터 바디 이름 변경
self.rewards.left_end_effector_position_tracking.params["asset_cfg"].body_names = ["ll_dg_ee"]
# ... (다른 왼쪽 팔 보상들도 동일하게 변경)

self.rewards.right_end_effector_position_tracking.params["asset_cfg"].body_names = ["rl_dg_ee"]
# ... (다른 오른쪽 팔 보상들도 동일하게 변경)


# 커맨드(commands) 설정에서 엔드 이펙터 바디 이름 변경
self.commands.left_ee_pose.body_name = "ll_dg_ee"
self.commands.right_ee_pose.body_name = "rl_dg_ee"
```

---

위와 같이 수정함으로써, 시뮬레이션 환경은 새로운 `openarm_tesollo_t1` 로봇을 성공적으로 스폰하고, 올바른 조인트와 링크를 참조하여 제어 및 상태 추적을 수행할 수 있게 됩니다.
