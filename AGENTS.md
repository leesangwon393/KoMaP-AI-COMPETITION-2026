## 1. Workspace Structure

프로젝트 루트는 이 AGENTS.md가 위치한 디렉토리이다. 문서의 모든 상대 경로는 프로젝트 루트를 기준으로 해석한다.

```text
프로젝트 루트/
├── AGENTS.md
│
├── Thirdparty/
│   ├── openpi/
│   └── ...
│
├── src/
│   └── *.py
│
├── script/
│   └── *.sh
│
├── resources/
│   └── ...
│
└── run/
    └── ...
```
이 외에 mart_ws 하위에 디렉토리를 임의로 만들지 않는다.

---

## 2. Directory Rules

### `Thirdparty/`

외부 repository 및 pretrained model 관련 파일을 저장한다.

예시:

```text
Thirdparty/
├── openpi/
├── lerobot/
├── groot/
└── ...
```

규칙:

* 외부 Git repository는 이 디렉토리에 clone한다.
* **Thirdparty 내부 코드는 절대 직접 수정하지 않는다.**
* 실험을 위해 기능을 변경해야 하는 경우 `src/`로 복사하여 수정을 진행한다
* pretrained weight가 특정 repository와 함께 관리되어야 하는 경우 해당 repository 내부의 공식 위치를 따른다.
* 외부 repo마다 venv나 conda 가상환경을 별도로 관리하지마라, 가상환경 추가가 필요하다면 사용자에게 물어보고 3개가 maximum이다. 3개 초과의 경우 사용자가 아니라 팀의 결정을 기다려야 한다
---

### `src/`

프로젝트에서 직접 작성하는 Python source code를 저장한다.

예시:

```text
src/
└── *.py
```

규칙:

* 프로젝트의 핵심 구현은 모두 `src/`에 작성한다.
* Thirdparty repository는 `src/`에서 import하여 사용한다.
* 실험별 하드코딩은 최대한 피한다.
* src 하위에 있는 smokerun 스크립트는 실행 후 삭제한다. (코드 내에 smokerun 분기를 만들지 않는다/)
* 일회용 코드는 src가 아닌 script/의 .sh로 작성해라
* 파일 수가 많아지지 않게 작업해라
* 하위 디렉토리를 임의로 만들지 말고 최대한 flat하게 작업해라
* 새로운 파일을 만들기 전에 기존 디렉토리를 읽어서 중복 파일을 확인해라

예:

```python
from thirdparty_library import SomeModel
```
---

### `script/`

실행용 shell script를 저장한다.

예시:

```text
script/
├── train.sh
├── eval.sh
├── inference.sh
├── teleop.sh
└── collect_data.sh
```

규칙:

* 반복적으로 사용하는 실행 명령을 저장한다.
* 주요 Python 로직을 `.sh` 안에 구현하지 않는다.
* 단순한 명령어로 실행할 수 있는 거는 .sh로 만들지 않는다.

---

### `resources/`

프로젝트 실행에 필요한 보조 파일을 저장한다. 공통 마운트 데이터(`mart_data` 등)를 심볼릭 링크로 연결하여 사용할 수 있다.

---

### `공통 마운트 (/workspace/*)`

서버 공용으로 마운트된 데이터, 시뮬레이션 환경, 에셋 경로이다.

| 공통 경로 | 용도 | 사용 규칙 |
|---|---|---|
| `/workspace/mart_data` | 공용 시연 데이터 (demonstration) | 대용량 데이터셋 원본이므로 `user/`로 복사하지 말고 직접 참조하거나 `resources/mart_data` 심볼릭 링크 활용 |
| `/workspace/cyclo_lab` | Isaac Lab 환경 | 공통 Python/시뮬레이션 환경이므로 직접 사용 (`user/`에 복제 금지) |
| `/workspace/assets` | 과제 공용 에셋 | 로봇, 매장, 상품 USD 및 텍스처 원본이므로 직접 참조 (`user/assets` 복사 금지) |
| `/workspace/challenge_scripts` | 과제 스크립트 배포 자리 | 2026-09-19 확인: 최신 공식 스크립트가 존재한다. 프로젝트 실행 기준은 아래 `humanoid-challenge-env` 체크아웃을 사용 |

규칙:

* **공통 마운트 내부 파일은 절대 직접 수정하지 않는다.**
* 공통 마운트에 존재하는 파일(에셋, parquet 데이터셋 등)을 `user/` 내부로 중복 복사하지 않는다.
* 필요 시 공통 마운트 절대 경로를 직접 참조하거나 `resources/` 아래에 심볼릭 링크를 생성하여 사용한다.
* 제공 과제 스크립트는 `/workspace/user/humanoid-challenge-env`의 Git 체크아웃을 기준 저장소로 관리 및 사용한다. 공통 마운트의 사본과 버전이 다를 수 있으므로 업데이트 시 기준 저장소의 커밋을 확인한다.

---

### `run/`

프로그램 실행으로 생성되는 모든 결과물을 저장한다.

권장 실행 디렉토리 이름은 아래 형식이다. `experiment-name` 은 사용자에게 물어본다.

```text
YYYYMMDD_HHMMSS_experiment-name
```

예:

```text
20260916_213000_pi05_g1_dex1_유연
20260917_104200_sonic_latency_test_성재승
```

---

## 3. Agent Rules

코딩 에이전트는 다음 규칙을 따른다.

1. `Thirdparty/` 내부 파일을 수정하지 않는다.
2. 프로젝트 구현은 `src/`에 작성한다.
3. 실행용 shell script는 `script/`에 작성한다.
4. 실행 결과는 반드시 `run/`에 저장한다.
5. 외부 repository가 필요하면 `Thirdparty/`에 clone한다.
6. 새로운 파일을 만들기 전에 기존 프로젝트 구조를 확인한다.
7. 기존 코드와 중복되는 기능을 새로 만들기 전에 기존 구현을 검색한다.
8. 임시 파일이나 디버그 결과를 repository root에 생성하지 않는다.
9. pretrained checkpoint 및 대용량 결과물을 Git에 commit하지 않는다.
10. 공통 마운트(`/workspace/mart_data`, `/workspace/cyclo_lab`, `/workspace/assets`) 내부 파일을 수정하지 않고, `user/` 내로 중복 복제하지 않는다.
11. 삭제가 default이고 보존이 option이다. 사용자의 지시가 없다면 기본적으로 삭제하고 보존이 필요하면 대화를 중단한 뒤 사용자에게 보존 여부를 허가 받아라


기본 원칙은 다음과 같다.

```text
공통 자료     → /workspace/ (mart_data, cyclo_lab, assets)
외부 코드     → Thirdparty/
우리 코드     → src/
실행 명령     → script/
보조 자료     → resources/
실행 결과     → run/
```

## 4. Coding Rules

절대로 지켜야 하는 규칙이다.

- 코드 작업할 때 과도한 캡슐화, 추상화는 가독성을 저해한다.
- 코드 작업할 때 최소수정으로 사용자의 요구사항을 달성해라.
- args나 os.environ, export처럼 외부에서 인자를 받는 것은 재현성을 저해하므로 사용자의 명시적인 요청이 있을 때만 반영해라.
- global constant는 가독성을 떨어뜨리고 불필요한 코드 라인을 증가시킨다.
- 함수 안의 함수는 지양한다.
- 껍데기 wrapper는 사용하지 않는다.
- 사용자의 요청을 정직하게 수행하고 만약 안되더라도 우회하지마라(대신 되물어라)
- 예외처리는 적용하지 마라 예외가 발생한 뒤에 수정해도 괜찮다.(한 번에 완벽한 코드를 작성하지 않는다.)
- 오류가 수정되면 예외처리를 하지 않아도 된다.(이미 강건한 코드이기 때문)
