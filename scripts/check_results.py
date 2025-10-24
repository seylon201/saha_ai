"""간단한 결과 확인 스크립트"""
import sys
import os
if os.name == 'nt':
    sys.path.append('C:/jangrim-lstm-prediction')
else:
    sys.path.append('/mnt/c/jangrim-lstm-prediction')

import numpy as np
from pathlib import Path
from src.config import MODELS_DIR

print("=" * 60)
print("Training Results Check")
print("=" * 60)

# Check training history
history_path = MODELS_DIR / 'training_history.npy'
if history_path.exists():
    history = np.load(history_path, allow_pickle=True).item()

    print("\nTraining History:")
    print(f"  Total Epochs: {len(history['loss'])}")
    print(f"  Final Train Loss: {history['loss'][-1]:.6f}")
    print(f"  Final Val Loss: {history['val_loss'][-1]:.6f}")
    print(f"  Final Train MAE: {history['mae'][-1]:.6f}")
    print(f"  Final Val MAE: {history['val_mae'][-1]:.6f}")

    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = np.min(history['val_loss'])
    print(f"\n  Best Performance (Epoch {best_epoch}):")
    print(f"    Val Loss: {best_val_loss:.6f}")
    print(f"    Val MAE: {history['val_mae'][best_epoch-1]:.6f}")
else:
    print("\nTraining history not found!")

# Check model file
model_path = MODELS_DIR / 'lstm_model_best.keras'
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024**2)
    print(f"\nModel File:")
    print(f"  Path: {model_path}")
    print(f"  Size: {size_mb:.2f} MB")
else:
    print("\nModel file not found!")

print("\n" + "=" * 60)
print("Check Complete!")
print("=" * 60)
