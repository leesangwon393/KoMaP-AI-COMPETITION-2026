# 최종 실행 파일 배치와 재현 계약

프로젝트 개요·성능·명령은 [README](../README.md), 구조와 학습 선택은 [모델링 문서](../docs/FINAL_MODELING_KO.md)를 참조하세요.

## 경로

| 경로 | 역할 |
|---|---|
| `src/` | 최종 공통 추론·앙상블·factory·평가 |
| `src/model{1,2,3}/` | 모델 생성·학습·추론 진입점과 모델별 보존 소스 |
| `resources/` | 데이터 경로·설정·환경·결과 메타데이터·라이선스 |
| `Thirdparty/weights/` | ImageNet 초기화 가중치(별도 공유, Git 제외) |
| `run/checkpoints/model{1,2,3}/best.pt` | 배포 체크포인트(별도 공유, Git 제외) |
| `run/inference/` | 새 확률·마스크·평가 결과 |
| `run/retraining/` | 새 학습 결과 |

별도 공유 ZIP(예: `A004092_weights.zip`)의 `02_program/models/modelN/best.pt`를 `run/checkpoints/modelN/best.pt`에, `02_program/Thirdparty/weights/*.pth`를 `Thirdparty/weights/`에 배치합니다. 파일을 자동 다운로드하지 않습니다.

## 입력·출력

- `settings.json`: `data_root`는 저장소 루트 기준 상대경로 또는 절대경로, `split`은 `valid`/`test`, `device`는 예를 들어 `cuda:0`.
- 데이터: `train/images`, `train/masks`, `valid/images`, `valid/masks`, `test/images`. 원본은 외부 경로를 직접 참조할 수 있습니다.
- 추론은 Validation20 또는 Test10을 확인하고 체크포인트 SHA256·epoch·config와 state dict를 검사합니다.
- 출력: `run/inference/{model1,model2,model3,ensemble}/{split}/{probabilities,masks}/`와 `metrics.json`.
- 확률은 원본 좌표의 float32 `[4,H,W]`, 채널 순서는 Primary Si / Al3Ni / Eutectic Si / Al입니다.
- 앙상블은 영상 SHA256·파일명·shape·채널 순서·확률 범위를 검사하고 고정 float32 30:35:35 순서로 합산합니다.

## 학습과 보존

세 모델은 서로 다른 내부 모듈을 사용하므로 독립 프로세스로 실행합니다. 새로운 학습 폴더가 이미 있으면 시작하지 않습니다. 체크포인트의 config와 소스 내부 state dict 구조는 보존합니다. `supervised_center` 등 과거 메타데이터 문구보다 최종 `full_context_supervision` 실행 플래그와 실제 학습 코드를 기준으로 감독 영역을 읽어야 합니다.

Model 1·2 공통 factory에서는 배포 recipe에 사용하지 않는 후보 옵션·보조 손실·증강을 제외했습니다. 기존 실험은 `experiments/` 및 Git 이력으로 추적합니다. 공개 초기화 가중치와 데이터가 없는 상태의 CPU 구조 검사를 실제 제출 체크포인트의 수치 재현으로 해석하지 않습니다.

기반 소스와 라이선스는 [licenses/SOURCES.txt](licenses/SOURCES.txt)에 있습니다. 기존 `mimu/`, 루트 스크립트, `experiments/`, `docs/`, `releases/`의 실행 경로는 유지합니다.
