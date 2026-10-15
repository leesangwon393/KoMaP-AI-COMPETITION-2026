# 문서 안내

현재 실행 대상은 `src/`의 최종 세 모델입니다. 과거 실험 자료는 기록 당시의 기준 모델·코드·장비에 대한 결과로 읽어야 합니다.

| 목적 | 문서 |
|---|---|
| 최종 모델과 실행 | [프로젝트 README](../README.md) |
| 보고서 기준 구조·학습·결과 | [최종 모델링](FINAL_MODELING_KO.md) |
| 증강 설정 비교와 완료 seed 결과 | [증강 실험](AUGMENTATION_EXPERIMENTS_KO.md) |
| 데이터 배치 | [데이터·재현 안내](DATA_AND_REPRODUCIBILITY_KO.md) |
| 가중치·입출력 계약 | [최종 실행 파일 배치](../resources/FINAL_MODELS.md) |
| 10월 8일까지의 실험 이력 | [전체 실험 스냅샷](experiment_snapshot_20261008/TOTAL_SUMMARY_KO.md) |
| 실험 계열·최신 완료 기록 | [실험 지도](EXPERIMENT_INDEX_KO.md) |
| 기존 R012–R044 계열 | [과거 실험 결과](EXPERIMENT_RESULTS_KO.md) |
| 과거 README | [2026-10-08 기록](README_HISTORY_20261008.md) |

`experiment_snapshot_20261008/`의 대기·실행 상태는 그 날짜의 스냅샷입니다. 이후 완료된 증강·밝기 실험은 최신 원자료 집계 문서를 우선합니다. 현재 README에서 R022를 최종 제출 모델로 안내하지 않으며, 기존 문서의 “현재 최고”, “진행 중”도 작성 시점의 표현입니다.

새 출력·체크포인트는 `run/`, 설정·환경·최종 메타데이터는 `resources/`에 둡니다. `docs/experiments/`에는 과거 실험의 Markdown 설명만 남겼습니다. 원 실행 코드·실험 설정·릴리스는 [보관 브랜치](https://github.com/leesangwon393/KoMaP-AI-COMPETITION-2026/tree/archive/experiments-before-cleanup-20261015)에서 확인할 수 있습니다. 모델별 `src/modelN/code/`는 최종 학습·추론의 의존성이므로 유지합니다.
