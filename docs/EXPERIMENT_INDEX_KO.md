# 실험 지도

최종 실행과 비교 실험을 구분한다. 현재 제출 모델은 `src/model1`, `src/model2`, `src/model3`이며 아래 과거 계열의 최고 점수나 대조군을 그대로 최종 모델의 성능으로 대체하지 않는다.

| 계열 | 코드 | 확인할 내용 |
|---|---|---|
| 최종 Model 1/2/3 | [src](../src/) | 보고서의 300/300/150epoch·30:35:35 구성 |
| R022 문맥·기본 구조 | [server_architecture_lab](../experiments/server_architecture_lab/README.md) | 과거 기준 모델과 구조 후보 |
| RRCU | [rrcu_scse_lab](../experiments/rrcu_scse_lab/README_KO.md) | decoder·공간/채널 gate의 재귀 경로 |
| 증강 강도·구성 | [r03_augmentation_lab](../experiments/r03_augmentation_lab/README_KO.md) | crop·기하 난수 정합을 고려한 증강 비교 |
| 밝기 축소 재검증 | [밝기 실험 코드](../experiments/r03_brightness_reproduction_lab_20261006/) | 별도 T4 환경에서 밝기 ±0.04 비교 |
| 접촉부·상 구분 | [phase_discrimination_lab](../experiments/phase_discrimination_lab/README_KO.md) | 접촉부 손실·상 구분 목적의 후보 |
| 후속 손실·연결성·LR | [followup_unet_lab](../experiments/followup_unet_lab/README_KO.md) | 동일 seed 대조와 후속 recipe |
| U²-Net | [u2net_server_lab](../experiments/u2net_server_lab/README_KO.md) | 중첩 RSU 기반 분할 |
| U²-Net attention | [u2net_scse_server_lab](../experiments/u2net_scse_server_lab/README_KO.md) | scSE 등 attention 후보 |
| scSE 위치·기전 | [scse_layer_lab_20261005](../experiments/scse_layer_lab_20261005/README_KO.md) | 단계별 attention 위치와 변형 |
| CNN/Transformer | [transformer_hybrid_lab_20261007](../experiments/transformer_hybrid_lab_20261007/README_KO.md) | bottleneck·pyramid·decoder attention 후보 |

## 완료 증강 실험의 최신 집계

L4 R03 28회와 T4 밝기 재검증 6회의 저장 JSON을 확인했다. 완결된 3seed 조건과 일부 seed만 확보된 조건을 구분한다. [성능표와 해석](AUGMENTATION_EXPERIMENTS_KO.md) · [run별 CSV](../resources/experiment_results/augmentation.csv)

평균 mIoU와 접촉부 오류의 상충, 밝기 축소의 장비·소스별 결과 차이가 있어 모든 모델에 같은 증강 변경을 적용하지 않는다. 원래 기본 증강이 모든 대안보다 최적이라는 근거로도 쓰지 않는다.

## U-Net + Transformer의 후속 완료 보고

2026-10-09에 보관된 서버 최종 보고는 B·HT_BOT·HT_PYR·HT_DEC16 × seed42/43/44의 12회 완료를 기록한다. 10월8일 스냅샷의 대기 상태보다 이후 자료이다. 다음 값은 보관된 서버 보고의 요약이며, 이 저장소 정리에서 원본 12개 checkpoint를 재평가한 결과는 아니다.

| 조건 | Single 평균(%) | D4 평균(%) | 같은 seed B 대비 D4 Δ(%p) | 개선 seed 수 |
|---|---:|---:|---:|---:|
| B | 79.2860 | 79.7511 | 기준 | — |
| HT_BOT | 79.2236 | 79.2055 | −0.5456 | 0/3 |
| HT_PYR | 79.2987 | 79.5578 | −0.1933 | 1/3 |
| HT_DEC16 | 79.3961 | 79.8133 | +0.0622 | 2/3 |

보고된 사전 승격 기준 +0.10%p에 미달하고 Eutectic IoU·skeleton recall 등 일부 품질이 악화되어 B를 대체하는 모델은 선정하지 않았다. 결과 보고에 따르면 checkpoint는 평가 후 삭제되었고 점수·설정·로그 등은 보존됐다. 이 계열을 최종 Model 1/2/3이나 별도로 보관한 HT_DEC16 seed44 checkpoint와 혼동하지 않는다.

## 과거 자료의 해석

- [전체 실험 표](experiment_snapshot_20261008/EXPERIMENTS.csv)는 역사적 실행·평가·seed 집계가 섞인 장부로, 행 수가 독립 학습 횟수는 아니다.
- 같은 seed·같은 환경의 대조군 대비 Δ를 우선한다. MPS·L4·T4·H100 또는 소스 버전이 다른 값을 하나의 모델 순위로 합치지 않는다.
- Validation으로 checkpoint·recipe를 선택한 결과는 독립 Test의 일반화 성능과 구분한다.
- 초기 후보를 학습 시작 시점의 task checkpoint에서 이어 학습했는지 공개 가중치에서 fresh 시작했는지 각 문서를 확인한다.
