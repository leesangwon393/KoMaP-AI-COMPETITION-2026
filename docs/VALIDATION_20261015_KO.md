# 최종 코드 검증 기록

최종 코드 비판 검토·제거 근거·비교 조건은 [코드 검토](CODE_REVIEW_20261015_KO.md)에 정리했다. 성능 값의 출처와 해석은 [모델링 문서](FINAL_MODELING_KO.md)를 따른다.

```bash
python -m pytest tests -q
```

**18 passed**. 확률 앙상블과 평가 지표 검사 15개, 세 모델의 실제 D4 타일 복원·체크포인트 검사·NPZ/RGB 출력 경로를 합성 모델로 실행하는 검사 3개다.

| 비교 | 확인 결과 |
|---|---|
| 세 모델 구성·초기화 | state dict 전체 tensor hash, 키, CPU RNG, 파라미터 수 동일 |
| 세 모델 CPU 출력 | 같은 64×64 입력에서 bitwise 동일 |
| 세 모델 손실 | 동일 logits/target에서 값·gradient 동일 |
| 세 모델 Train 패치 | 8개 샘플의 영상·라벨·원본 선택·RNG 동일 |
| 세 모델 타일 추론 | 합성 영상의 실제 D4 확률 동일 |
| Model 1/2 공통 학습 루프 | 합성 모델·dataset 1epoch의 체크포인트와 손실 기록 동일; 시간 필드 제외 |
| 진입점·정적 확인 | 학습/추론 6개 import, Model 1/2 설정 구성, 구문·전역 이름·주요 문서 링크 확인 |

실제 배포 가중치와 CUDA를 사용한 전체 Validation/Test 재추론 또는 재학습을 수행한 기록은 아니다. Model 1/2 구조 비교에서는 가중치 다운로드 없이 합성 backbone 초기화를 사용했다. 고정 recipe 및 체크포인트 SHA256·epoch·config 검사는 정상 실행 경로에 유지한다.
