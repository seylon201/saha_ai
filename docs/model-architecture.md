# LSTM 모델 아키텍처 문서

**프로젝트**: 장림유수지 LSTM 수위 예측 모델
**모델 버전**: v1.0 (Simple LSTM)
**작성일**: 2025년 10월 14일

---

## 1. 모델 개요

### 1.1 목적
장림 유수지의 수위를 30분 후 예측하여 방재 시설 자동 제어에 활용

### 1.2 입출력
- **입력**: 과거 60분간의 유수지 A, B 수위 데이터
- **출력**: 30분 후의 유수지 A, B 수위 예측값

### 1.3 모델 타입
**Sequence-to-Point LSTM (Many-to-One)**
- 시계열 시퀀스(60분) → 단일 예측값(30분 후)

---

## 2. 모델 아키텍처

### 2.1 전체 구조

```
입력 데이터 (60분 × 2개 특성)
         ↓
┌────────────────────┐
│   LSTM Layer 1     │  128 units, return_sequences=True
│   (128 units)      │
└────────────────────┘
         ↓
┌────────────────────┐
│   Dropout (0.2)    │  과적합 방지
└────────────────────┘
         ↓
┌────────────────────┐
│   LSTM Layer 2     │  64 units
│   (64 units)       │
└────────────────────┘
         ↓
┌────────────────────┐
│   Dropout (0.2)    │  과적합 방지
└────────────────────┘
         ↓
┌────────────────────┐
│   Dense (32)       │  ReLU activation
└────────────────────┘
         ↓
┌────────────────────┐
│   Dense (2)        │  출력층 (유수지 A, B)
└────────────────────┘
         ↓
    예측 수위 (2개)
```

### 2.2 레이어 상세

| 레이어 | 타입 | 출력 Shape | 파라미터 수 | 설명 |
|--------|------|-----------|------------|------|
| Input | InputLayer | (None, 60, 2) | 0 | 60분 × 2특성 |
| LSTM 1 | LSTM | (None, 60, 128) | 67,072 | 128 유닛, 시퀀스 반환 |
| Dropout 1 | Dropout | (None, 60, 128) | 0 | 20% 드롭아웃 |
| LSTM 2 | LSTM | (None, 64) | 49,408 | 64 유닛 |
| Dropout 2 | Dropout | (None, 64) | 0 | 20% 드롭아웃 |
| Dense 1 | Dense | (None, 32) | 2,080 | ReLU 활성화 |
| Dense 2 | Dense | (None, 2) | 66 | 출력층 |

**총 파라미터**: 118,626개 (463.38 KB)

### 2.3 파라미터 계산

#### LSTM Layer 1 (67,072개)
```
파라미터 = 4 × (input_dim + hidden_dim + 1) × hidden_dim
         = 4 × (2 + 128 + 1) × 128
         = 4 × 131 × 128
         = 67,072
```

#### LSTM Layer 2 (49,408개)
```
파라미터 = 4 × (128 + 64 + 1) × 64
         = 4 × 193 × 64
         = 49,408
```

---

## 3. 하이퍼파라미터

### 3.1 모델 구조 파라미터
```python
LSTM_UNITS = [128, 64]     # LSTM 레이어별 유닛 수
DROPOUT_RATE = 0.2         # 드롭아웃 비율
DENSE_UNITS = 32           # Dense 레이어 유닛 수
OUTPUT_DIM = 2             # 출력 차원 (유수지 A, B)
```

### 3.2 학습 파라미터
```python
SEQUENCE_LENGTH = 60       # 입력 시퀀스 길이 (분)
PREDICTION_HORIZON = 30    # 예측 시점 (분)
BATCH_SIZE = 32            # 배치 크기
LEARNING_RATE = 0.001      # 학습률
EPOCHS = 50                # 최대 에포크
```

### 3.3 옵티마이저 및 손실 함수
- **옵티마이저**: Adam (learning_rate=0.001)
- **손실 함수**: MSE (Mean Squared Error)
- **평가 지표**: MAE (Mean Absolute Error)

---

## 4. 데이터 전처리

### 4.1 정규화
**Min-Max Scaling**
```
normalized_value = (x - min) / (max - min)
```

각 특성(유수지 A, B)별로 독립적으로 스케일링 적용

### 4.2 시퀀스 생성
```python
for i in range(60, len(data) - 30):
    # 입력: i-60 ~ i 시점 (60분)
    X[i] = data[i-60:i]
    # 출력: i+30 시점 (30분 후)
    y[i] = data[i+30]
```

### 4.3 데이터셋 분할
- **Train**: 70% (30,176 샘플)
- **Validation**: 20% (8,622 샘플)
- **Test**: 10% (4,312 샘플)

**시계열 분할**: 시간 순서 유지 (Shuffling 없음)

---

## 5. 학습 전략

### 5.1 콜백 함수

#### Early Stopping
```python
monitor='val_loss'
patience=10
restore_best_weights=True
```
- 검증 손실이 10 에포크 동안 개선되지 않으면 조기 종료
- 최고 성능 가중치 복원

#### Model Checkpoint
```python
monitor='val_loss'
save_best_only=True
```
- 검증 손실이 개선될 때마다 모델 저장

#### ReduceLROnPlateau
```python
monitor='val_loss'
factor=0.5
patience=5
min_lr=1e-6
```
- 5 에포크 동안 개선이 없으면 학습률 50% 감소

### 5.2 과적합 방지
1. **Dropout (0.2)**: 각 LSTM 레이어 후 적용
2. **Early Stopping**: 검증 손실 기반 조기 종료
3. **Validation Set**: 20% 검증 데이터로 성능 모니터링

---

## 6. 모델 입출력 사양

### 6.1 입력 데이터 형식
```python
Input Shape: (batch_size, 60, 2)
- batch_size: 배치 크기 (32)
- 60: 시퀀스 길이 (과거 60분)
- 2: 특성 수 (유수지 A, B)

Data Type: float32
Range: [0, 1] (정규화됨)
```

### 6.2 출력 데이터 형식
```python
Output Shape: (batch_size, 2)
- batch_size: 배치 크기
- 2: 예측값 (유수지 A, B의 30분 후 수위)

Data Type: float32
Range: [0, 1] (정규화됨)
```

### 6.3 역정규화
```python
real_value = normalized_value × (max - min) + min
```

---

## 7. 성능 목표

### 7.1 목표 지표
| 지표 | 목표값 | 설명 |
|------|--------|------|
| RMSE | < 5cm | 제곱 평균 제곱근 오차 |
| MAE | < 3cm | 평균 절대 오차 |
| 응답 시간 | < 1초 | 실시간 예측 |
| 정확도 | > 90% | 전체 정확도 |

### 7.2 평가 기준
- **실시간성**: 30분 전 예측으로 충분한 대응 시간 확보
- **정확성**: 5cm 이내 오차로 정밀한 수위 제어
- **안정성**: 극단 상황에서도 일관된 예측 성능

---

## 8. 모델 한계 및 개선 방향

### 8.1 현재 한계 (Phase 1)
1. **단순 입력**: 유수지 A, B 수위만 사용
2. **단일 예측 시점**: 30분 후만 예측
3. **외부 요인 미고려**: 강수량, 펌프 상태 등 미포함

### 8.2 개선 계획

#### Phase 2: 다변량 입력
```
추가 특성:
- 외수위 (SYSTEM.LT_104)
- 펌프 운전 상태 (게이트 펌프 1~6)
- 펌프 전류값
- 외수위 변화율
```

#### Phase 3: 다중 시점 예측
```
예측 시점:
- 10분 후
- 30분 후
- 60분 후
```

#### Phase 4: 기상 데이터 통합
```
외부 데이터:
- 강수량 (실시간 + 예보)
- 기온, 습도, 풍속
```

---

## 9. 기술 스택

### 9.1 프레임워크
- **TensorFlow**: 2.19.0
- **Keras**: TensorFlow 내장

### 9.2 주요 라이브러리
```python
numpy==1.24.3
pandas==2.0.3
```

### 9.3 하드웨어 요구사항
- **메모리**: 최소 4GB RAM
- **저장공간**: 최소 2GB
- **CPU/GPU**: CPU 사용 (GPU 선택적)

---

## 10. 모델 파일

### 10.1 저장 형식
- **형식**: Keras 모델 (.keras)
- **경로**: `models/lstm_model_best.keras`
- **크기**: 약 463 KB

### 10.2 로딩 방법
```python
from tensorflow import keras
model = keras.models.load_model('models/lstm_model_best.keras')
```

---

**작성자**: Claude Code
**버전**: 1.0
**다음 업데이트**: 학습 완료 및 성능 평가 후
