# 최종 모델링과 실험 근거

이 문서는 최종 보고서의 구조·선택 결과와 실제 배포 코드를 연결합니다. 개발 이력의 수치는 보고서 기록이며, 현재 저장소 정리 과정에서 새 학습을 수행한 수치가 아닙니다.

## 1. 세 모델의 입력과 구조

공통 입력은 [0,1]의 grayscale `N×1×448×448`입니다. 의미 경로는 3채널 반복 후 ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`로 정규화합니다. Model 1·2의 raw detail은 이 ImageNet 정규화를 적용하기 전 grayscale을 사용합니다.

### Model 1: ConvNeXt-Tiny U-Net + Raw Detail

문맥 특징과 미세한 상세 정보를 함께 전달합니다. ConvNeXt-Tiny의 4단계 특징(96/192/384/768채널)을 U-Net decoder의 skip concatenation으로 복원하고 scSE를 적용합니다. 상위 256채널 decoder 특징에는 global context를 잔차로 더합니다.

별도 stride-2·16채널 grayscale stem 뒤에서 depthwise 3×3·5×5 residual refiner를 적용합니다. Decoder1은 64채널 의미 특징과 16채널 detail을 **연결**하고, Decoder0은 복원된 64채널 특징과 1채널 raw grayscale을 연결합니다. 최종 32채널 특징에서 1×1 head가 네 클래스 logits를 출력합니다. Detail을 최종 logits에 직접 더하는 구조가 아닙니다.

- [실제 forward](../src/model1/code/mimu/convnext_context.py)
- [최종 decoder 옵션·초기화](../src/model_factory.py)
- [고정 설정](../resources/model1/config.json): full448 감독, SiLU decoder, decoder residual, global context, BN momentum 0.04.

### Model 2: ResNet101 U-Net + Semantic-Gated Detail

의미 정보를 이용해 상세 특징의 기여를 조절합니다. ResNet101 U-Net/scSE에서 최종 32채널 의미 특징 `F`를 얻고, 별도 stride-2·16채널 detail stem과 3×3·5×5 residual refiner로 `D`를 생성합니다.

`G = sigmoid(Conv1×1(resize(F)))`로 gate를 만들고 `D×G`를 projection·보간하여 의미 특징에 잔차로 더합니다. 실제 결합은 `F + residual_fusion(resize(detail_to_semantic(D×G)))`입니다. `residual_fusion.weight`를 0으로 초기화하며 별도 학습 scalar α를 사용하지 않습니다.

- [Semantic gate·forward](../src/model2/code/mimu/gated_shape.py)
- [의미 경로](../src/model2/code/mimu/imagenet_context.py)
- [고정 설정](../resources/model2/config.json): full448 감독, distinct-image batches, squared Dice.

### Model 3: Full U²-Net + ECA

중첩 U 구조의 RSU로 여러 해상도 특징을 처리합니다. 6개 encoder·5개 decoder의 11개 outer RSU residual branch에 ECA를 적용한 뒤 residual 입력을 더합니다. 6개 side output을 resize·concatenate하고 1×1 convolution으로 융합합니다. 출력 융합과 별도 deep supervision 손실은 구분하며 최종 모델은 후자를 사용하지 않습니다.

입력 정규화에는 ImageNet mean/std를 사용하지만 **사전학습 가중치는 사용하지 않습니다.** 학습 감독은 중앙 224×224이고 평가 출력은 원본 좌표로 복원합니다.

- [RSU·ECA 위치](../src/model3/code/komap_phase/rsu_models.py)
- [고정 설정](../resources/model3/config.json)

## 2. 학습·증강

공통 조건은 seed42, batch4, epoch당 420개 패치, AdamW lr=3×10⁻⁴, weight decay=10⁻⁴, cosine scheduler입니다. 손실 계수는 CE 0.5 + Dice 0.25 + Lovasz 0.25로, Dice·Lovasz의 클래스 가중치와 집계는 각 모델 구현을 따릅니다. 모델별 손실을 완전히 동일한 식이라고 표현하지 않습니다.

| 조건 | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| 총 / 선택 epoch | 300 / 172 | 300 / 184 | 150 / 135 |
| 감독 영역 | 전체448 | 전체448 | 중앙224 |
| Dice | batch·공간 합산 | 영상별 집계 기반 squared Dice 변경 | batch·공간 합산 |
| Lovasz 클래스 | 전체4개 | 정답에 존재 | 정답에 존재 |
| 추가 조건 | 경계 샘플링·Cutout·GC | batch 내 다른 원본4개 | 중첩 RSU·ECA |

### 코드로 확인한 증강 순서

| 모델 | 순서 |
|---|---|
| Model 1 | 패치 추출 → 같은 회전·반전 → 광도 증강 → batch에서 Cutout → 모델 내부 정규화 |
| Model 2 | 패치 추출 → 같은 회전·반전 → 광도 증강 → 모델 내부 정규화 |
| Model 3 | 패치 추출 → 광도 증강 → 같은 회전·반전 → 모델 내부 정규화 |

회전은 0/90/180/270도 균등 선택, 좌우 반전은 p=0.5입니다. 광도 증강은 영상에만 적용하며, 밝기·대비는 p=0.5로 함께 적용합니다. 패치 평균 μ를 기준으로 `(I−μ)c+μ+b`, `c∈[0.9,1.1]`, `b∈[−0.08,0.08]`입니다. 감마는 p=0.3·지수0.9~1.1, 평균0 Gaussian noise는 p=0.15·σ=0.01이고 결과를 [0,1]로 제한합니다.

Model 1의 Cutout은 **패치마다 p=0.5**, 입력 안에 완전히 들어가는 64×64 영역을 무작위 선택해 그 패치의 평균으로 채우며 정답은 유지합니다. 원래 난수 사용과 변환 순서를 유지합니다. [Model 1 sampler](../src/model1/code/mimu/pro6000_context.py) · [Model 2 sampler](../src/model2/code/mimu/pro6000_context.py) · [Model 3 sampler](../src/model3/code/komap_phase/data.py) · [Cutout](../src/model_factory.py)

## 3. 개발 과정에서 선택한 조건

보고서에 기록된 변화량은 각 변경 직전의 모델별 최고 설정 대비 **Validation D4 차이**입니다. 서로 다른 기준 설정에서 측정했으므로 합산하지 않으며 반복 seed로 모듈의 독립 효과를 검증한 결과가 아닙니다.

| 변경 | Model 1 Δ(%p) | Model 2 Δ(%p) | 채택 |
|---|---:|---:|---|
| 경계 샘플링 | +0.1399 | −1.7719 | Model 1 |
| Gradient centralization | +0.0873 | −0.1219 | Model 1 |
| 전체448 감독 | +0.0968 | +0.0120 | Model 1·2 |
| 전체 클래스 Lovasz | +0.0049 | −0.3307 | Model 1 |
| Batch 내 서로 다른 원본 | −0.0828 | +0.0726 | Model 2 |
| Cutout | +0.1140 | −0.0434 | Model 1 |
| Squared Dice | −0.0461 | +0.2025 | Model 2 |

실제 설정·연결 경로는 저장소 코드로 확인할 수 있습니다. 위 개발 변화량의 실험별 원본 로그는 이번 공개 코드 묶음에 포함되지 않아 보고서 기록과 별도 원자료를 구분합니다. 다중 seed의 증강 비교 원자료는 [별도 R03 실험](AUGMENTATION_EXPERIMENTS_KO.md)에 제공합니다.

비중30:35:35를 고정한 학습 기간 비교에서는 다음 결과가 보고됐습니다. 각 실행의 단일 시점 Validation 최고 checkpoint를 D4로 평가했으며, 단일 모델의 epoch 효과를 분리한 실험은 아닙니다.

| Model 1/2/3의 총 epoch | Validation D4 mIoU |
|---|---:|
| 150 / 150 / 150 | 80.8611% |
| 300 / 300 / 300 | 80.8189% |
| 300 / 300 / 150 | **81.0183%** |

## 4. 추론과 앙상블

Reflect padding112를 두고 입력448의 중앙224를 stride112로 배치합니다. D4는 네 회전 × 좌우 반전 유무의 8시점입니다. 각 logits의 softmax 후 좌표를 역변환하고 중앙 확률을 취해 평균하며, 겹치는 타일도 평균합니다.

6개 후보의 2·3·4모델 조합을 양의 5% 간격 비중으로 비교하고 추가 모델을 포함한 8개 후보에서도 점검했습니다. 검토 격자의 최고는 30:35:35였습니다. 추가 모델1%의 개선 약0.000382%p는 실행 부담과 함께 고려해 최종 구성에 포함하지 않았습니다.

`Pfinal = 0.30 Pmodel1 + 0.35 Pmodel2 + 0.35 Pmodel3`

모든 영상·클래스에 같은 비중을 적용하고 마지막에 한 번만 argmax합니다. 연속 비중 공간 전체의 최적값이나 모든 영상의 개선을 보장하지 않습니다.

| 비교 구성 | Validation mIoU |
|---|---:|
| 최고 단일 Model 2 | 80.3861% |
| Model 1/2 = 40:60¹ | 80.8450% |
| 세 모델 동일 비중¹ | 80.9997% |
| 최종30:35:35 재현 | **81.0183%** |

¹ 탐색 당시 저장 확률 기준. 최종 재현값은 81.01828805454941%, 탐색 기준값은 81.01829207313666%로 경계1픽셀 차이입니다.

같은 checkpoint에서 단일 시점→D4는 Model 1: 79.6670→80.3762%, Model 2: 79.6006→80.3861%, Model 3: 80.0073→80.2360%였습니다.

## 5. 남은 오류와 재현 범위

최종 앙상블은 Model 2 대비 17/20장 개선, 3장 하락했습니다. Eutectic Si가 가장 많이 개선됐으나 IoU61.6317%로 가장 어렵습니다. 사후 경계 분석에서 경계 영역 오류율11.43%, 내부1.24%였고 전체 오류의89.53%가 경계 영역에 있었습니다.

같은 Validation을 반복 사용한 선택 편향과 미확인 시편 독립성이 남습니다. 전체 GPU 재추론이나 반복 학습을 이 저장소 정리 과정에서 수행한 것은 아닙니다. 체크포인트별 수치 재현은 별도 가중치·데이터·환경을 준비해 확인해야 하며, 작은 연산 차이가 경계 argmax에 영향을 줄 수 있습니다.
