"""예측 결과 시각화 스크립트"""
import sys
import os
if os.name == 'nt':
    sys.path.append('C:/jangrim-lstm-prediction')
else:
    sys.path.append('/mnt/c/jangrim-lstm-prediction')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 저장만
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import MODELS_DIR, RESULTS_DIR

print("=" * 60)
print("예측 결과 시각화")
print("=" * 60)

# Step 1: 평가 결과 로드
print("\n[Step 1] 평가 결과 로딩...")
results_path = MODELS_DIR / 'evaluation_results.npy'

if not results_path.exists():
    print(f"  ✗ 평가 결과 파일이 없습니다: {results_path}")
    print("  먼저 evaluate_model.py를 실행해주세요.")
    exit(1)

results = np.load(results_path, allow_pickle=True).item()
y_test_real = results['y_test_real']
y_pred_real = results['y_pred_real']

print(f"  ✓ 로드 완료")
print(f"  샘플 수: {len(y_test_real)}")

# Step 2: 시각화 - 전체 예측
print("\n[Step 2] 전체 예측 결과 시각화...")

# 처음 500개만 표시 (너무 많으면 보기 어려움)
n_samples = min(500, len(y_test_real))

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 유수지 A
axes[0].plot(y_test_real[:n_samples, 0], label='실제 수위', linewidth=1.5, alpha=0.7)
axes[0].plot(y_pred_real[:n_samples, 0], label='예측 수위', linewidth=1.5, alpha=0.7)
axes[0].set_title('유수지 A 수위 예측 (처음 500개 샘플)', fontsize=14)
axes[0].set_xlabel('샘플 인덱스')
axes[0].set_ylabel('수위 (m)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 유수지 B
axes[1].plot(y_test_real[:n_samples, 1], label='실제 수위', linewidth=1.5, alpha=0.7, color='orange')
axes[1].plot(y_pred_real[:n_samples, 1], label='예측 수위', linewidth=1.5, alpha=0.7, color='green')
axes[1].set_title('유수지 B 수위 예측 (처음 500개 샘플)', fontsize=14)
axes[1].set_xlabel('샘플 인덱스')
axes[1].set_ylabel('수위 (m)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_path1 = RESULTS_DIR / 'figures' / '02_prediction_timeseries.png'
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"  ✓ 저장: {output_path1}")
plt.close()

# Step 3: 산점도 (Scatter Plot)
print("\n[Step 3] 산점도 생성...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 유수지 A
axes[0].scatter(y_test_real[:, 0], y_pred_real[:, 0], alpha=0.3, s=10)
axes[0].plot([y_test_real[:, 0].min(), y_test_real[:, 0].max()],
             [y_test_real[:, 0].min(), y_test_real[:, 0].max()],
             'r--', linewidth=2, label='완벽한 예측')
axes[0].set_title('유수지 A - 실제 vs 예측', fontsize=14)
axes[0].set_xlabel('실제 수위 (m)')
axes[0].set_ylabel('예측 수위 (m)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 유수지 B
axes[1].scatter(y_test_real[:, 1], y_pred_real[:, 1], alpha=0.3, s=10, color='orange')
axes[1].plot([y_test_real[:, 1].min(), y_test_real[:, 1].max()],
             [y_test_real[:, 1].min(), y_test_real[:, 1].max()],
             'r--', linewidth=2, label='완벽한 예측')
axes[1].set_title('유수지 B - 실제 vs 예측', fontsize=14)
axes[1].set_xlabel('실제 수위 (m)')
axes[1].set_ylabel('예측 수위 (m)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_path2 = RESULTS_DIR / 'figures' / '03_scatter_plot.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"  ✓ 저장: {output_path2}")
plt.close()

# Step 4: 오차 분포
print("\n[Step 4] 오차 분포 시각화...")

errors_A = y_pred_real[:, 0] - y_test_real[:, 0]
errors_B = y_pred_real[:, 1] - y_test_real[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 유수지 A 오차
axes[0].hist(errors_A, bins=50, alpha=0.7, edgecolor='black')
axes[0].axvline(0, color='r', linestyle='--', linewidth=2, label='오차 = 0')
axes[0].set_title(f'유수지 A 오차 분포\n평균: {errors_A.mean():.4f}m, 표준편차: {errors_A.std():.4f}m', fontsize=12)
axes[0].set_xlabel('오차 (예측 - 실제) [m]')
axes[0].set_ylabel('빈도')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 유수지 B 오차
axes[1].hist(errors_B, bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[1].axvline(0, color='r', linestyle='--', linewidth=2, label='오차 = 0')
axes[1].set_title(f'유수지 B 오차 분포\n평균: {errors_B.mean():.4f}m, 표준편차: {errors_B.std():.4f}m', fontsize=12)
axes[1].set_xlabel('오차 (예측 - 실제) [m]')
axes[1].set_ylabel('빈도')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_path3 = RESULTS_DIR / 'figures' / '04_error_distribution.png'
plt.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"  ✓ 저장: {output_path3}")
plt.close()

print("\n" + "=" * 60)
print("시각화 완료!")
print("=" * 60)
print(f"\n생성된 그래프:")
print(f"  1. {output_path1}")
print(f"  2. {output_path2}")
print(f"  3. {output_path3}")
