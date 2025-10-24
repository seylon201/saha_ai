"""데이터 전처리 파이프라인 - 샘플 버전 (최근 1개월)
빠른 테스트를 위해 최근 1개월 데이터만 사용
"""
import sys
sys.path.append('/mnt/c/jangrim-lstm-prediction')

from src.data.loader import load_integrated_data
from src.data.preprocessing import handle_missing_values, normalize_data
from src.data.sequence import create_sequences, split_train_val_test
from src.config import (
    INTEGRATED_DATA_PATH,
    TARGET_COLUMNS,
    TARGET_NAMES,
    SEQUENCE_LENGTH,
    PROCESSED_DATA_DIR,
    TRAIN_RATIO,
    VAL_RATIO
)
import numpy as np

print("=" * 60)
print("데이터 전처리 파이프라인 (샘플 - 최근 1개월)")
print("=" * 60)

# Step 1: 데이터 로드 및 샘플링
print("\n[Step 1] 데이터 로딩 및 샘플링...")
df = load_integrated_data(INTEGRATED_DATA_PATH)
print(f"  전체 데이터: {df.shape}")

# 최근 1개월만 추출 (약 43,000개)
n_samples = 30 * 24 * 60  # 30일 * 24시간 * 60분
df_sample = df.tail(n_samples).copy()
print(f"  샘플 데이터: {df_sample.shape} (최근 {n_samples:,}분)")

# Step 2: 타겟 변수만 추출
print("\n[Step 2] 타겟 변수 추출...")
df_target = df_sample[TARGET_COLUMNS].copy()
df_target.columns = TARGET_NAMES
print(f"  컬럼: {TARGET_NAMES}")

# Step 3: 결측값 처리
print("\n[Step 3] 결측값 처리...")
missing_before = df_target.isnull().sum().sum()
df_target = handle_missing_values(df_target)
missing_after = df_target.isnull().sum().sum()
print(f"  처리 전: {missing_before}개 → 처리 후: {missing_after}개")

# Step 4: 정규화
print("\n[Step 4] 데이터 정규화...")
df_normalized, scaler = normalize_data(df_target, TARGET_NAMES)
print(f"  정규화 완료 (Min-Max)")

# Scaler 저장
import pickle
scaler_path = PROCESSED_DATA_DIR / 'scaler_sample.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  Scaler 저장: {scaler_path}")

# Step 5: 시퀀스 생성
print(f"\n[Step 5] 시계열 시퀀스 생성...")
print(f"  입력: 과거 {SEQUENCE_LENGTH}분")
print(f"  예측: 30분 후")

X, y = create_sequences(
    data=df_normalized,
    feature_columns=TARGET_NAMES,
    target_columns=TARGET_NAMES,
    sequence_length=SEQUENCE_LENGTH,
    prediction_horizon=30
)

print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# Step 6: 데이터셋 분할
print(f"\n[Step 6] 데이터셋 분할...")
(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_train_val_test(
    X, y, TRAIN_RATIO, VAL_RATIO
)

print(f"  Train: {X_train.shape[0]:,}개")
print(f"  Val:   {X_val.shape[0]:,}개")
print(f"  Test:  {X_test.shape[0]:,}개")

# Step 7: 저장
print(f"\n[Step 7] 저장...")
np.save(PROCESSED_DATA_DIR / 'X_train_sample.npy', X_train)
np.save(PROCESSED_DATA_DIR / 'y_train_sample.npy', y_train)
np.save(PROCESSED_DATA_DIR / 'X_val_sample.npy', X_val)
np.save(PROCESSED_DATA_DIR / 'y_val_sample.npy', y_val)
np.save(PROCESSED_DATA_DIR / 'X_test_sample.npy', X_test)
np.save(PROCESSED_DATA_DIR / 'y_test_sample.npy', y_test)

print(f"  저장 완료: {PROCESSED_DATA_DIR}/")

print("\n" + "=" * 60)
print("전처리 완료!")
print("=" * 60)
