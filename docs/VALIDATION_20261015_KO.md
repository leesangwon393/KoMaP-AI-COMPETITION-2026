# 최종 코드 정리 검증 — 2026-10-15

팀원 브랜치 `codex/final-model123-code`의 커밋 `070fd0b3925b9c33df64613a7ed56f82870427ef`를 기준으로 정리했다.

## 변경

- 바이트 단위로 동일했던 세 `infer.py`의 본문을 공통 `src/inference.py`로 모았다. 각 모델은 계속 독립 프로세스로 실행한다.
- `model_factory.py`를 475줄에서 90줄로 줄였다. 최종 R052/R073에서 사용하지 않는 후보 구조·보조 손실·증강과 사용하지 않는 import를 제외했다.
- 모델별 `src/modelN/code/`와 `resources/modelN/config.json`은 기준 커밋과 동일하다. 체크포인트·공개 가중치 해시와 최종 학습 조건은 유지했다.
- 앙상블의 원본 영상·채널·크기·확률 검사를 명시적인 오류로 처리하고 NPZ를 닫도록 정리했다. 고정 30/35/35 float32 합산 순서는 유지했다.
- 팔레트를 공통 평가 모듈에 모았다. 최종 재현값과 탐색 기준값을 구분해 기록했다.
- 과거 코드·실험 경로를 보존하고 README와 문서 진입점을 최종 보고서 중심으로 재구성했다.

## 확인한 항목

| 검사 | 결과와 범위 |
|---|---|
| 확률 결합·RGB·평가 계약 | `tests/test_final_pipeline.py` 15개 통과 |
| 잘못된 입력 거부 | 원본 해시·채널 순서·크기·음수·NaN·dtype·확률합·비중 검사 |
| Test10 / Validation20 출력 | 합성 확률 파일로 마스크·출력 경로·image-macro IoU 및 부재 클래스 규칙 검사 |
| Cutout | 패치별 적용·64×64·패치 평균 대체·원본/정답 유지 검사 |
| Model 1/2 factory 정리 전후 | 합성 backbone 초기화에서 state dict 키·텐서·생성 후 CPU RNG·strict load·64×64 평가 출력이 동일 |
| 파라미터 수 | Model 1: 32,458,571 / Model 2: 51,614,459 / Model 3: 44,051,485 |
| Model 3 실행 | scratch 모델의 CPU 64×64 입력에서 유한한 `[1,4,64,64]` 출력 확인 |
| 진입점 | 세 모델의 학습·추론 import와 Model 1/2의 학습 configure 확인 |
| 파일 검사 | `src/` Python 구문, `resources/` JSON, 새 문서의 로컬 링크, Git whitespace 검사 |

Model 1/2의 factory 비교에서는 네트워크 다운로드나 실제 공개 가중치가 필요하지 않도록 테스트용 backbone 초기화를 사용했다. 이는 정리 과정의 구성·초기화 순서 보존 검사이며 배포 checkpoint의 정확도 재현이 아니다. 실제 가중치의 SHA256 검사는 정상 실행 경로에 그대로 남아 있다.

## 수행하지 않은 항목

새 300/300/150epoch 학습, 실제 최종 checkpoint를 사용한 Validation20/Test10 전체 CUDA 추론 및 보고서 mIoU 재측정은 수행하지 않았다. 이를 위해서는 별도 제공 가중치·데이터와 호환 환경이 필요하다. README의 81.0183%는 최종 보고서·배포 설정의 기존 결과이다.
