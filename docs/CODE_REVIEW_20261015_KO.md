# 최종 코드 비판 검토와 정리

첫 정리는 루트 실험 파일을 제거했지만 `src/modelN/code/` 안의 실험 프레임워크까지 정리하지 못했다. 실제 학습·추론 경로를 다시 추적하여 아래 문제를 수정했다.

| 문제 | 영향 | 수정 |
|---|---|---|
| 미사용 backbone·구조 후보가 import 체인에 남음 | 최종 모델 범위와 의존성이 불명확 | HRNet, 다른 ResNet/ConvNeXt 초기화, SegNet, U2NETP adapter, 외부 timm/SMP/NASA 가중치 경로 등 실험용 생성자 제거 |
| 실험 큐·리포트·구 CLI가 내부 코드에 남음 | 과거 `runs/`·macOS 큐·존재하지 않는 실험 파일 참조 | 최종 호출 함수만 남기고 launcher·과거 평가 main·리포트 생성 제거 |
| 같은 학습·평가 코드 복사와 전역 monkey patch | 변경 누락 및 숨은 결합 | 공통 `training.py`, `evaluation.py`, `tiling.py`, `checkpoint_io.py`; 모델·dataset·loss·증강·gradient 처리를 명시적인 인자로 전달 |
| 꺼진 손실·증강 분기가 최종 코드에 남음 | 설정을 잘못 바꾸면 미지원 경로 실행 | Cutout·GC·모델별 Dice/Lovasz 등 채택 조건 유지, CutMix·mixup·jitter·Tversky·보조 contrast/deep/connection/OCR 등 미채택 구현 제거 |
| 입력·체크포인트 검증에 `assert` 사용 | Python `-O`에서 일부 검사가 사라짐 | 추론 SHA256·epoch·config·확률 및 공통 마스크 검사를 명시적 예외로 변경; 재학습 출력 보호도 `FileExistsError` 적용 |

## 규모와 배치

`src/` Python 파일은 **67개·7,198줄 → 50개·3,198줄**로 줄였다(이 검토 직전 main 대비). 모델별 레시피와 데이터 경로는 `resources/`, 실행 결과는 `run/`, 실험 기록은 `docs/`에 둔다. 데이터·가중치·개인 deliverables는 추가하지 않았다.

동명 `mimu` 패키지는 Model 1·2의 서로 다른 Dice 집계·샘플링 구현 때문에 모델별로 유지한다. 하나의 Python 프로세스에서 두 패키지를 섞지 않고 기존처럼 독립 프로세스로 실행한다. 공통 모듈만 `src/`로 올렸으며, 이 제약을 없애기 위한 별도 패키징 재설계는 하지 않았다.

## 제거하지 않은 부분

- **Model 1/2/3의 실제 구조·초기화 순서·샘플링·손실 차이:** 발표용 그림보다 실제 forward와 체크포인트 구조가 기준이다.
- **고정 JSON recipe:** 체크포인트 config와 정확히 비교하므로 과거 표현이나 비활성 메타데이터가 있어도 임의 수정하지 않았다.
- **일부 0값 로그·RNG 필드와 Identity 모듈:** 기존 학습 저장 형식과 state dict·수치 연산을 유지하는 호환 부분이다. 미지원 후보 구현을 뜻하지 않는다.
- **U²-Net upstream 소스:** 라이선스·원 출처를 유지하며 `vendor_u2net.py`는 수정하지 않았다. 포함된 upstream 정의 중 사용하지 않는 것을 제거하는 대신 파생 adapter를 최종 U2_ECA로 제한했다.
- **실제 사용되는 contact-band 진단:** Model 3의 평가 코드에서 호출하므로 학습용 보조 손실과 구분해 유지했다.

## 검증과 한계

- 전체 테스트 **18개 통과**: 확률 앙상블·지표·오류 파일 검사 15개, 세 모델의 실제 D4 타일 경로를 사용하는 추론 내보내기 검사 3개.
- 정리 전후 세 모델의 **state dict 키·모든 tensor hash·초기화 후 CPU RNG·64×64 CPU 출력·파라미터 수가 일치**.
- 모델별 실제 손실의 값과 logits gradient가 동일 fixture에서 **bitwise 일치**. Model 1/2의 다른 Dice 집계도 유지.
- 세 모델별 **8개 학습 패치·라벨·선택 원본·RNG**, 합성 영상의 실제 **D4 확률**이 정리 전후 일치.
- Model 1/2는 작은 합성 모델·dataset으로 기존/새 학습 루프를 1epoch 실행해 **model·optimizer·scheduler·RNG·손실 기록이 일치**(실행 시간 필드 제외).
- 학습·추론 진입점 6개 import, Model 1/2 recipe 구성, Python 구문 및 미정의 전역 이름 확인.

모델 비교는 공개 초기화 가중치 읽기를 대체한 합성 backbone 초기화이며 실제 배포 체크포인트의 GPU 정확도 재현은 아니다. 추론 테스트도 실제 타일·저장 경로에 작은 합성 모델을 연결한다. 전체 CUDA Valid/Test 재추론과 300/300/150epoch 재학습은 수행하지 않았다.

정리 전 main은 Git 이력 `ffb26ba7416c45d846945949faa6d3d42d70e738`에서 확인할 수 있다. 더 이전의 전체 실험 코드는 `archive/experiments-before-cleanup-20261015` 브랜치에 보존되어 있다.
