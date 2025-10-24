"""학습 히스토리 분석 스크립트"""
import sys
import os
if os.name == 'nt':
    sys.path.append('C:/jangrim-lstm-prediction')
else:
    sys.path.append('/mnt/c/jangrim-lstm-prediction')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import MODELS_DIR, RESULTS_DIR

print("=" * 60)
print("학습 히스토리 분석")
print("=" * 60)

# Step 1: 학습 히스토리 로드
print("\n[Step 1] 학습 히스토리 로딩...")
history_path = MODELS_DIR / 'training_history.npy'

if not history_path.exists():
    print(f"  ✗ 학습 히스토리 파일이 없습니다: {history_path}")
    print("  학습이 완료될 때까지 기다려주세요.")
    exit(1)

history = np.load(history_path, allow_pickle=True).item()
print(f"  ✓ 로드 완료")
print(f"  총 에포크: {len(history['loss'])}")

# Step 2: 학습 곡선 시각화
print("\n[Step 2] 학습 곡선 시각화...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss 그래프
axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('모델 손실(Loss) 변화', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# MAE 그래프
axes[1].plot(history['mae'], label='Train MAE', linewidth=2)
axes[1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].set_title('평균 절대 오차(MAE) 변화', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = RESULTS_DIR / 'figures' / '05_training_history.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  ✓ 저장: {output_path}")
plt.close()

# Step 3: 학습 통계
print("\n[Step 3] 학습 통계 분석...")

best_epoch = np.argmin(history['val_loss']) + 1
best_val_loss = np.min(history['val_loss'])
final_train_loss = history['loss'][-1]
final_val_loss = history['val_loss'][-1]

print(f"\n  최종 성능:")
print(f"    Train Loss: {final_train_loss:.6f}")
print(f"    Val Loss: {final_val_loss:.6f}")
print(f"    Train MAE: {history['mae'][-1]:.6f}")
print(f"    Val MAE: {history['val_mae'][-1]:.6f}")

print(f"\n  최고 성능 (Epoch {best_epoch}):")
print(f"    Val Loss: {best_val_loss:.6f}")
print(f"    Val MAE: {history['val_mae'][best_epoch-1]:.6f}")

# 과적합 체크
overfitting_gap = final_val_loss - final_train_loss
if overfitting_gap > 0.0001:
    print(f"\n  ⚠ 과적합 가능성: Val Loss - Train Loss = {overfitting_gap:.6f}")
else:
    print(f"\n  ✓ 과적합 없음: Val Loss - Train Loss = {overfitting_gap:.6f}")

# 조기 종료 여부
total_epochs = len(history['loss'])
if total_epochs < 50:
    print(f"\n  ✓ 조기 종료 발동: {total_epochs} 에포크에서 종료")
else:
    print(f"\n  전체 에포크 완료: {total_epochs} 에포크")

print("\n" + "=" * 60)
print("분석 완료!")
print("=" * 60)
