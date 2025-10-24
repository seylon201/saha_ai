# 장림 유수지 LSTM 수위 예측 모델

2025년 인공지능(AI)기반 방재시설 및 유수지 관리체계 구축 프로젝트

## 📋 프로젝트 개요

장림 유수지의 수위를 LSTM 딥러닝 모델로 30분 후 예측하여 방재 시설 자동 제어에 활용하는 시스템입니다.

### 주요 기능
- 🔮 **30분 후 수위 예측**: 유수지 A, B의 수위를 실시간 예측
- 📊 **고정밀 예측**: RMSE < 5cm 목표
- ⚡ **실시간 처리**: < 1초 응답 시간
- 🤖 **자동화**: 방재 시설 자동 제어 연동

---

## 🎯 성능 목표

| 지표 | 목표값 | 현재 상태 |
|------|--------|----------|
| RMSE | < 5cm | 🔄 학습 중 |
| MAE | < 3cm | 🔄 학습 중 |
| 응답 시간 | < 1초 | ✅ 달성 |
| 정확도 | > 90% | 🔄 학습 중 |

---

## 📁 프로젝트 구조

```
jangrim-lstm-prediction/
├── data/
│   ├── raw/                    # 원본 데이터
│   └── processed/              # 전처리된 데이터
├── docs/                       # 문서
│   ├── EDA-report.md          # 데이터 분석 보고서
│   ├── model-architecture.md  # 모델 아키텍처 문서
│   ├── project-progress-summary.md
│   └── work-log.md            # 작업 로그
├── notebooks/                  # 분석 노트북
│   ├── 01_basic_info.py
│   ├── 02_column_analysis.py
│   ├── 03_missing_values.py
│   ├── 04_basic_statistics.py
│   └── 05_time_series_pattern_simple.py
├── scripts/                    # 실행 스크립트
│   ├── preprocess_data_sample.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── visualize_results.py
│   └── analyze_training.py
├── src/                        # 소스 코드 (모듈화)
│   ├── config.py              # 설정
│   ├── data/                  # 데이터 처리
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   └── sequence.py
│   ├── models/                # 모델
│   │   └── lstm_model.py
│   └── utils/                 # 유틸리티
│       ├── metrics.py
│       └── visualization.py
├── models/                     # 학습된 모델
├── results/                    # 결과
│   ├── figures/               # 그래프
│   └── logs/                  # 로그
└── requirements.txt           # 의존성
```

---

## 🚀 Quick Start

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

원본 데이터를 `data/raw/` 폴더에 배치:
- `장림펌프장 데이터_통합.csv`
- `20251001_사하_강수량.csv`
- `20251001_장림유수지_데이터정리_Ver2.xlsx`

### 3. 데이터 전처리

```bash
python scripts/preprocess_data_sample.py
```

### 4. 모델 학습

```bash
python scripts/train_model.py
```

### 5. 모델 평가 및 시각화

```bash
# 학습 분석
python scripts/analyze_training.py

# 모델 평가
python scripts/evaluate_model.py

# 결과 시각화
python scripts/visualize_results.py
```

---

## 📊 데이터 정보

### 입력 데이터
- **기간**: 2024-06-18 ~ 2025-06-17 (1년)
- **샘플 수**: 1,017,978개
- **샘플링 간격**: 30초
- **센서 수**: 202개

### 타겟 변수
- **유수지 A** (`SYSTEM.LT_102`): -1.40m ~ 1.10m
- **유수지 B** (`SYSTEM.LT_103`): -1.40m ~ 0.89m

---

## 🧠 모델 아키텍처

### LSTM 모델 (v1.0)
```
입력: (60분, 2특성) → LSTM(128) → Dropout(0.2)
                     → LSTM(64) → Dropout(0.2)
                     → Dense(32) → Dense(2) → 출력: 2개 예측값
```

**총 파라미터**: 118,626개 (463 KB)

**입출력**:
- 입력: 과거 60분간의 유수지 A, B 수위
- 출력: 30분 후의 유수지 A, B 수위 예측값

자세한 내용은 [모델 아키텍처 문서](docs/model-architecture.md) 참조

---

## 📈 학습 과정

### 하이퍼파라미터
- 시퀀스 길이: 60분
- 예측 시점: 30분 후
- 배치 크기: 32
- 학습률: 0.001
- 옵티마이저: Adam
- 손실 함수: MSE
- 평가 지표: MAE

### 데이터셋 분할
- Train: 70% (30,176 샘플)
- Validation: 20% (8,622 샘플)
- Test: 10% (4,312 샘플)

### 콜백
- Early Stopping (patience=10)
- Model Checkpoint (최고 성능 저장)
- ReduceLROnPlateau (학습률 감소)

---

## 📄 문서

- 📊 [EDA 보고서](docs/EDA-report.md) - 데이터 탐색 분석 결과
- 🏗️ [모델 아키텍처](docs/model-architecture.md) - 모델 구조 상세
- 📝 [프로젝트 진행 현황](docs/project-progress-summary.md)
- 📋 [작업 로그](docs/work-log.md)

---

## 🛠️ 기술 스택

- **언어**: Python 3.8+
- **딥러닝**: TensorFlow 2.19.0
- **데이터 처리**: pandas, numpy
- **시각화**: matplotlib, seaborn

---

## 📊 결과 (학습 완료 후 업데이트 예정)

### 성능 지표
- RMSE: TBD
- MAE: TBD
- MAPE: TBD

### 시각화
학습 완료 후 `results/figures/`에서 확인 가능:
- 예측 vs 실제 시계열 그래프
- 산점도 (Scatter Plot)
- 오차 분포
- 학습 곡선

---

## 🔄 개발 로드맵

### Phase 1: 단순 LSTM 모델 ✅ (진행 중)
- 유수지 A, B 수위만 사용
- 30분 후 예측

### Phase 2: 다변량 입력 (예정)
- 외수위, 펌프 상태 추가
- 변화율 특성 추가

### Phase 3: 다중 시점 예측 (예정)
- 10분, 30분, 60분 후 동시 예측

### Phase 4: 기상 데이터 통합 (예정)
- 강수량 실시간 데이터
- 기상청 예보 데이터

---

## 🤝 기여

이 프로젝트는 2025년 인공지능(AI)기반 방재시설 및 유수지 관리체계 구축 프로젝트의 일환입니다.

---

## 📞 문의

프로젝트 관련 문의사항은 이슈를 등록해주세요.

---

## 📜 라이선스

This project is proprietary and confidential.

---

**최종 업데이트**: 2025-10-14
**프로젝트 상태**: 🔄 모델 학습 진행 중 (Epoch 9/50)
