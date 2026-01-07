# Isaac Lab Task 생성 및 디버깅 과정 요약

이 문서는 새로운 'approach' task를 생성하고, 로봇을 교체하며 발생했던 주요 오류들과 그 해결 과정을 정리합니다.

---

### 1. 환경 등록 오류 (`gymnasium.error.NameNotFound`)

-   **문제**: `approach` task를 새로 만들고 실행했을 때, `gymnasium`이 `Isaac-Approach-OpenArm-Bi` 환경을 찾지 못하는 에러가 계속 발생했습니다.
-   **원인**: `openarm`은 `isaaclab` 외부의 독립적인 파이썬 패키지이므로, `isaaclab`이 실행될 때 `openarm` 패키지 내부의 task들이 자동으로 등록되지 않았습니다. `gym.register` 코드가 포함된 `__init__.py` 파일이 실행되지 않았기 때문입니다.
-   **해결 과정**:
    1.  **가설 1**: `pip install -e` 재설치를 통해 패키지 정보를 갱신하면 해결될 것이라 예상했으나 실패.
    2.  **가설 2**: `isaaclab_tasks`의 최상위 `__init__.py`에 `import openarm.tasks`를 추가하여 `openarm` 패키지를 로드하려고 시도했으나, 여전히 `gym.register`가 실행되지 않았습니다. (이후 `import_packages` 함수의 미묘한 동작 방식 문제로 추정)
    3.  **최종 해결**: `openarm` 패키지의 task들이 시작되는 지점인 `openarm/openarm/tasks/__init__.py` 파일에, 자동 탐지(`import_packages`)에만 의존하지 않고 새로 만든 `approach` task의 설정 파일을 **명시적으로 직접 import** 하도록 코드를 추가했습니다. 이로써 `gym.register` 코드가 반드시 실행되도록 보장했습니다.
        -   **수정 파일**: `openarm_isaac_lab/source/openarm/openarm/tasks/__init__.py`
        -   **추가된 코드**:
            ```python
            # Explicitly import the new 'approach' task config to ensure registration
            import openarm.tasks.manager_based.openarm_manipulation.bimanual.approach.config
            ```

---

### 2. 모듈 경로 오류 (`ModuleNotFoundError: No module named 'source'`)

-   **문제**: 환경 등록 문제가 해결된 후, `source`라는 모듈을 찾을 수 없다는 에러가 발생했습니다.
-   **원인**: `openarm` 패키지 내부의 여러 파일(`joint_pos_env_cfg.py`, `assets/openarm_bimanual.py` 등)에서 `from source.openarm...`과 같이 잘못된 절대 경로로 다른 모듈을 `import`하고 있었습니다. `pip`으로 패키지를 설치하면 `openarm`이 최상위 경로가 되므로, 그 앞에 `source`가 붙으면 안 됩니다.
-   **해결**: 에러가 발생하는 파일들을 찾아다니며, `from source.openarm...`으로 시작하는 모든 `import` 구문을 `from openarm...` 또는 올바른 상대 경로(`from ..`)로 수정했습니다.
    -   **수정 파일 1**: `.../approach/config/joint_pos_env_cfg.py`
    -   **수정 파일 2**: `.../assets/openarm_bimanual.py`

---

### 3. 로봇 바디(Body) 탐색 오류 (`ValueError: Not all regular expressions are matched! ... ll_dg_ee: []`)

-   **문제**: 시뮬레이션이 시작되고 로봇을 불러오는 과정에서, `ll_dg_ee`라는 이름의 바디를 찾을 수 없다는 에러가 발생했습니다.
-   **원인**: 사용자가 USD 파일에 추가한 `ll_dg_ee`와 `rl_dg_ee`가 물리 속성이 없는 순수한 `Xform` prim이었기 때문입니다. Isaac Lab의 `find_bodies()` 함수는 물리적 실체(Rigid Body)가 있는 링크만 '바디'로 인식하므로, 이름만 있는 `Xform`은 찾을 수 없었습니다.
-   **해결**: 엔드 이펙터의 위치 추적을 위해, 실제로 로봇에 존재하는 물리 바디의 이름을 사용하도록 설정을 변경했습니다. URDF 분석 결과, 손목에 해당하는 `openarm_left_link7`과 `openarm_right_link7`이 가장 적합한 바디임을 확인하고 이 이름으로 교체했습니다.
    -   **수정 파일**: `.../approach/config/joint_pos_env_cfg.py`
    -   **수정 내용**: `rewards` 및 `commands` 설정에서 `body_names`를 `ll_dg_ee` -> `openarm_left_link7`으로, `rl_dg_ee` -> `openarm_right_link7`으로 변경.

---

### 4. 성능 및 물리 엔진 경고 (로딩 시간 및 `Patch buffer overflow`)

-   **문제**: 시뮬레이션 로딩에 매우 오랜 시간이 걸리고, `PhysX error: Patch buffer overflow detected`라는 경고가 대량으로 발생했습니다.
-   **원인**: 환경 설정에 `num_envs = 4096`으로 설정되어, 4096개의 복잡한 로봇 환경을 동시에 생성하려 했기 때문입니다. 이는 과도한 메모리 및 CPU/GPU 자원을 요구하여 로딩 시간을 지연시키고 물리 엔진의 내부 버퍼 한계를 초과하게 만듭니다.
-   **해결**: 디버깅 및 개발 초기 단계에서는 환경 수를 대폭 줄여서 테스트하는 것이 효율적입니다. `num_envs`를 `64`와 같은 작은 값으로 줄일 것을 제안했습니다.
    -   **수정 제안 파일**: `.../approach/approach_env_cfg.py`
    -   **수정 제안 내용**: `self.scene.num_envs = 4096` -> `self.scene.num_envs = 64`
