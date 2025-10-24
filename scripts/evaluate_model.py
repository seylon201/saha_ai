"""LSTM 모델 평가 스크립트"""
import sys
import os
if os.name == 'nt':
    sys.path.append('C:/jangrim-lstm-prediction')
else:
    sys.path.append('/mnt/c/jangrim-lstm-prediction')

import numpy as np
from pathlib import Path
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from src.utils.metrics import calculate_metrics, print_metrics
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

print("=" * 60)
print("LSTM 모델 평가")
print("=" * 60)

# Step 1: 테스트 데이터 로드
print("\n[Step 1] 테스트 데이터 로딩...")
X_test = np.load(PROCESSED_DATA_DIR / 'X_test_sample.npy')
y_test = np.load(PROCESSED_DATA_DIR / 'y_test_sample.npy')

print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")

# Step 2: 학습된 모델 로드
print("\n[Step 2] 학습된 모델 로딩...")
model_path = MODELS_DIR / 'lstm_model_best.keras'

if not model_path.exists():
    print(f"  ✗ 모델 파일을 찾을 수 없습니다: {model_path}")
    print("  학습이 완료될 때까지 기다려주세요.")
    exit(1)

model = keras.models.load_model(model_path)
print(f"  ✓ 모델 로드 완료: {model_path}")

# Step 3: 예측 수행
print("\n[Step 3] 예측 수행...")
y_pred = model.predict(X_test, verbose=0)
print(f"  예측 완료: {y_pred.shape}")

# Step 4: 성능 평가
print("\n[Step 4] 성능 지표 계산...")

# 전체 성능
metrics_all = calculate_metrics(y_test, y_pred)
print_metrics(metrics_all, "전체")

# 유수지 A 개별 평가
metrics_A = calculate_metrics(y_test[:, 0], y_pred[:, 0])
print_metrics(metrics_A, "유수지 A")

# 유수지 B 개별 평가
metrics_B = calculate_metrics(y_test[:, 1], y_pred[:, 1])
print_metrics(metrics_B, "유수지 B")

# Step 5: Scaler로 역변환 (실제 수위로 변환)
print("\n[Step 5] 정규화 역변환...")
with open(PROCESSED_DATA_DIR / 'scaler_sample.pkl', 'rb') as f:
    scaler = pickle.load(f)

y_test_real = scaler.inverse_transform(y_test)
y_pred_real = scaler.inverse_transform(y_pred)

print(f"  실제 수위 범위:")
print(f"    유수지 A: {y_test_real[:, 0].min():.2f}m ~ {y_test_real[:, 0].max():.2f}m")
print(f"    유수지 B: {y_test_real[:, 1].min():.2f}m ~ {y_test_real[:, 1].max():.2f}m")

# 실제 수위 기준 성능
metrics_real = calculate_metrics(y_test_real, y_pred_real)
print(f"\n  실제 수위 기준 성능:")
print(f"    RMSE: {metrics_real['RMSE']:.4f}m ({metrics_real['RMSE']*100:.2f}cm)")
print(f"    MAE: {metrics_real['MAE']:.4f}m ({metrics_real['MAE']*100:.2f}cm)")

# 목표 달성 여부
target_rmse_cm = 5.0  # 5cm
actual_rmse_cm = metrics_real['RMSE'] * 100

if actual_rmse_cm <= target_rmse_cm:
    print(f"\n  ✓ 목표 달성! RMSE {actual_rmse_cm:.2f}cm <= {target_rmse_cm}cm")
else:
    print(f"\n  ✗ 목표 미달성: RMSE {actual_rmse_cm:.2f}cm > {target_rmse_cm}cm")

# Step 6: 결과 저장
print("\n[Step 6] 평가 결과 저장...")
results = {
    'metrics_normalized': metrics_all,
    'metrics_real': metrics_real,
    'metrics_A': metrics_A,
    'metrics_B': metrics_B,
    'y_test': y_test,
    'y_pred': y_pred,
    'y_test_real': y_test_real,
    'y_pred_real': y_pred_real
}

results_path = MODELS_DIR / 'evaluation_results.npy'
np.save(results_path, results)
print(f"  저장 완료: {results_path}")

print("\n" + "=" * 60)
print("평가 완료!")
print("=" * 60)
