"""평가 지표 유틸리티"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(y_true, y_pred) -> dict:
    """평가 지표 계산"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


def print_metrics(metrics: dict, dataset_name: str = "Test"):
    """평가 지표 출력"""
    print(f"\n=== {dataset_name} Set Metrics ===")
    for metric, value in metrics.items():
        if metric == 'MAPE':
            print(f"{metric}: {value:.2f}%")
        else:
            print(f"{metric}: {value:.4f}")
