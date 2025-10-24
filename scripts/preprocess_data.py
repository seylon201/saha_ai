"""데이터 전처리 파이프라인 - Simple Version
타겟 변수(유수지 A, B)만 사용하여 시퀀스 생성
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
import pandas as pd

print("=" * 60)
print("데이터 전처리 파이프라인 시작")
print("=" * 60)

# Step 1: 데이터 로드
print("\n[Step 1] 데이터 로딩...")
df = load_integrated_data(INTEGRATED_DATA_PATH)
print(f"  원본 데이터: {df.shape}")

# Step 2: 타겟 변수만 추출
print("\n[Step 2] 타겟 변수 추출...")
df_target = df[['시간'] + TARGET_COLUMNS].copy()
df_target.columns = ['timestamp'] + TARGET_NAMES
print(f"  추출 데이터: {df_target.shape}")
print(f"  컬럼: {df_target.columns.tolist()}")

# Step 3: 결측값 처리
print("\n[Step 3] 결측값 처리...")
print(f"  처리 전 결측값: {df_target[TARGET_NAMES].isnull().sum().sum()}개")
df_target = handle_missing_values(df_target[TARGET_NAMES])
print(f"  처리 후 결측값: {df_target.isnull().sum().sum()}개")

# Step 4: 정규화 (Min-Max Scaling)
print("\n[Step 4] 데이터 정규화...")
df_normalized, scaler = normalize_data(df_target, TARGET_NAMES)
print(f"  정규화 범위: [0, 1]")
print(f"  Scaler 저장...")

# Scaler 정보 저장
import pickle
scaler_path = PROCESSED_DATA_DIR / 'scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  저장 위치: {scaler_path}")

# Step 5: 시퀀스 생성
print(f"\n[Step 5] 시계열 시퀀스 생성...")
print(f"  시퀀스 길이: {SEQUENCE_LENGTH}분 (입력)")
print(f"  예측 시점: 30분 후")

X, y = create_sequences(
    data=df_normalized,
    feature_columns=TARGET_NAMES,
    target_columns=TARGET_NAMES,
    sequence_length=SEQUENCE_LENGTH,
    prediction_horizon=30
)

print(f"  생성된 시퀀스:")
print(f"    X shape: {X.shape}  # (샘플수, 시퀀스길이, 특성수)")
print(f"    y shape: {y.shape}  # (샘플수, 타겟수)")

# Step 6: 데이터셋 분할
print(f"\n[Step 6] 데이터셋 분할...")
(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_train_val_test(
    X, y, TRAIN_RATIO, VAL_RATIO
)

print(f"  Train: {X_train.shape[0]:,}개 ({TRAIN_RATIO*100:.0f}%)")
print(f"  Val:   {X_val.shape[0]:,}개 ({VAL_RATIO*100:.0f}%)")
print(f"  Test:  {X_test.shape[0]:,}개 ({(1-TRAIN_RATIO-VAL_RATIO)*100:.0f}%)")

# Step 7: 저장
print(f"\n[Step 7] 전처리 데이터 저장...")
np.save(PROCESSED_DATA_DIR / 'X_train.npy', X_train)
np.save(PROCESSED_DATA_DIR / 'y_train.npy', y_train)
np.save(PROCESSED_DATA_DIR / 'X_val.npy', X_val)
np.save(PROCESSED_DATA_DIR / 'y_val.npy', y_val)
np.save(PROCESSED_DATA_DIR / 'X_test.npy', X_test)
np.save(PROCESSED_DATA_DIR / 'y_test.npy', y_test)

print(f"  저장 완료: {PROCESSED_DATA_DIR}/")

print("\n" + "=" * 60)
print("전처리 완료!")
print("=" * 60)
