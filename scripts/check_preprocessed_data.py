"""전처리된 데이터 확인"""
import sys
sys.path.append('/mnt/c/jangrim-lstm-prediction')

import numpy as np
from pathlib import Path
import pickle

PROCESSED_DIR = Path('/mnt/c/jangrim-lstm-prediction/data/processed')

print("=" * 60)
print("전처리된 데이터 확인")
print("=" * 60)

# 저장된 파일 목록
print("\n[저장된 파일 목록]")
files = sorted(PROCESSED_DIR.glob('*'))
for f in files:
    size_mb = f.stat().st_size / (1024**2)
    print(f"  - {f.name:30s} ({size_mb:.2f} MB)")

# 데이터 로드
print("\n[데이터 로드 및 Shape 확인]")
X_train = np.load(PROCESSED_DIR / 'X_train_sample.npy')
y_train = np.load(PROCESSED_DIR / 'y_train_sample.npy')
X_val = np.load(PROCESSED_DIR / 'X_val_sample.npy')
y_val = np.load(PROCESSED_DIR / 'y_val_sample.npy')
X_test = np.load(PROCESSED_DIR / 'X_test_sample.npy')
y_test = np.load(PROCESSED_DIR / 'y_test_sample.npy')

print(f"\nTrain Set:")
print(f"  X_train: {X_train.shape} - (샘플수, 시퀀스길이, 특성수)")
print(f"  y_train: {y_train.shape} - (샘플수, 타겟수)")

print(f"\nValidation Set:")
print(f"  X_val: {X_val.shape}")
print(f"  y_val: {y_val.shape}")

print(f"\nTest Set:")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")

# 데이터 샘플 확인
print("\n[데이터 샘플 확인 - Train Set 첫 번째]")
print(f"\n입력 (X_train[0]):")
print(f"  Shape: {X_train[0].shape} (60분 시퀀스 × 2개 특성)")
print(f"  첫 5개 타임스텝:")
print(X_train[0][:5])

print(f"\n출력 (y_train[0]):")
print(f"  Shape: {y_train[0].shape} (2개 타겟: reservoir_A, reservoir_B)")
print(f"  값: {y_train[0]}")

# 정규화 범위 확인
print("\n[정규화 범위 확인]")
print(f"  X_train - min: {X_train.min():.4f}, max: {X_train.max():.4f}")
print(f"  y_train - min: {y_train.min():.4f}, max: {y_train.max():.4f}")

# Scaler 정보
print("\n[Scaler 정보]")
with open(PROCESSED_DIR / 'scaler_sample.pkl', 'rb') as f:
    scaler = pickle.load(f)
print(f"  원본 최소값: {scaler.min_}")
print(f"  원본 최대값: {scaler.max_}")
print(f"  스케일 범위: {scaler.scale_}")

print("\n" + "=" * 60)
print("확인 완료!")
print("=" * 60)
