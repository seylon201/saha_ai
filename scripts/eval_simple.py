"""Simple evaluation script"""
import sys
import os
if os.name == 'nt':
    sys.path.append('C:/jangrim-lstm-prediction')
else:
    sys.path.append('/mnt/c/jangrim-lstm-prediction')

import numpy as np
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

print("=" * 60)
print("Model Evaluation")
print("=" * 60)

# Load test data
X_test = np.load(PROCESSED_DATA_DIR / 'X_test_sample.npy')
y_test = np.load(PROCESSED_DATA_DIR / 'y_test_sample.npy')
print(f"\nTest data: {X_test.shape}")

# Load model
model = keras.models.load_model(MODELS_DIR / 'lstm_model_best.keras')
print("Model loaded")

# Predict
y_pred = model.predict(X_test, verbose=0)
print(f"Prediction completed: {y_pred.shape}")

# Calculate metrics (normalized)
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\nNormalized Metrics:")
print(f"  RMSE: {rmse:.6f}")
print(f"  MAE: {mae:.6f}")

# Load scaler and inverse transform
with open(PROCESSED_DATA_DIR / 'scaler_sample.pkl', 'rb') as f:
    scaler = pickle.load(f)

y_test_real = scaler.inverse_transform(y_test)
y_pred_real = scaler.inverse_transform(y_pred)

# Real metrics
rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mae_real = mean_absolute_error(y_test_real, y_pred_real)

print(f"\nReal Metrics (meters):")
print(f"  RMSE: {rmse_real:.4f}m ({rmse_real*100:.2f}cm)")
print(f"  MAE: {mae_real:.4f}m ({mae_real*100:.2f}cm)")

target_rmse_cm = 5.0
actual_rmse_cm = rmse_real * 100

if actual_rmse_cm <= target_rmse_cm:
    print(f"\nTARGET ACHIEVED! RMSE {actual_rmse_cm:.2f}cm <= {target_rmse_cm}cm")
else:
    print(f"\nTarget not met: RMSE {actual_rmse_cm:.2f}cm > {target_rmse_cm}cm")

# Save results
results = {
    'y_test': y_test,
    'y_pred': y_pred,
    'y_test_real': y_test_real,
    'y_pred_real': y_pred_real,
    'rmse': rmse,
    'mae': mae,
    'rmse_real': rmse_real,
    'mae_real': mae_real
}

np.save(MODELS_DIR / 'evaluation_results.npy', results)
print(f"\nResults saved to {MODELS_DIR / 'evaluation_results.npy'}")

print("\n" + "=" * 60)
print("Evaluation Complete!")
print("=" * 60)
