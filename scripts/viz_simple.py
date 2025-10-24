"""Simple visualization script"""
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
from src.config import MODELS_DIR, RESULTS_DIR

print("Generating visualizations...")

# Load results
results = np.load(MODELS_DIR / 'evaluation_results.npy', allow_pickle=True).item()
y_test_real = results['y_test_real']
y_pred_real = results['y_pred_real']

# Plot 1: Time series (first 500 samples)
n_samples = min(500, len(y_test_real))
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(y_test_real[:n_samples, 0], label='Actual', linewidth=1.5, alpha=0.7)
axes[0].plot(y_pred_real[:n_samples, 0], label='Predicted', linewidth=1.5, alpha=0.7)
axes[0].set_title('Reservoir A - Water Level Prediction', fontsize=14)
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Water Level (m)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(y_test_real[:n_samples, 1], label='Actual', linewidth=1.5, alpha=0.7, color='orange')
axes[1].plot(y_pred_real[:n_samples, 1], label='Predicted', linewidth=1.5, alpha=0.7, color='green')
axes[1].set_title('Reservoir B - Water Level Prediction', fontsize=14)
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Water Level (m)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output1 = RESULTS_DIR / 'figures' / '02_prediction_timeseries.png'
plt.savefig(output1, dpi=150, bbox_inches='tight')
print(f"Saved: {output1}")
plt.close()

# Plot 2: Scatter plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_test_real[:, 0], y_pred_real[:, 0], alpha=0.3, s=10)
axes[0].plot([y_test_real[:, 0].min(), y_test_real[:, 0].max()],
             [y_test_real[:, 0].min(), y_test_real[:, 0].max()],
             'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_title('Reservoir A - Actual vs Predicted', fontsize=14)
axes[0].set_xlabel('Actual Water Level (m)')
axes[0].set_ylabel('Predicted Water Level (m)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test_real[:, 1], y_pred_real[:, 1], alpha=0.3, s=10, color='orange')
axes[1].plot([y_test_real[:, 1].min(), y_test_real[:, 1].max()],
             [y_test_real[:, 1].min(), y_test_real[:, 1].max()],
             'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_title('Reservoir B - Actual vs Predicted', fontsize=14)
axes[1].set_xlabel('Actual Water Level (m)')
axes[1].set_ylabel('Predicted Water Level (m)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output2 = RESULTS_DIR / 'figures' / '03_scatter_plot.png'
plt.savefig(output2, dpi=150, bbox_inches='tight')
print(f"Saved: {output2}")
plt.close()

print("Visualization complete!")
