"""시각화 유틸리티"""
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, save_path: str = None):
    """학습 히스토리 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Model Loss')

    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].set_title('Model MAE')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_predictions(y_true, y_pred, target_names: list, save_path: str = None):
    """예측 결과 시각화"""
    n_targets = y_true.shape[1]
    fig, axes = plt.subplots(n_targets, 1, figsize=(12, 4*n_targets))

    if n_targets == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, target_names)):
        ax.plot(y_true[:, i], label='Actual', alpha=0.7)
        ax.plot(y_pred[:, i], label='Predicted', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Water Level (m)')
        ax.legend()
        ax.set_title(f'{name} - Prediction vs Actual')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
