"""프로젝트 설정 파일"""
from pathlib import Path
import os

# 경로 설정 (Windows와 Linux 모두 지원)
if os.name == 'nt':  # Windows
    PROJECT_ROOT = Path("C:/jangrim-lstm-prediction")
else:  # Linux/WSL
    PROJECT_ROOT = Path("/mnt/c/jangrim-lstm-prediction")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# 데이터 파일 경로
INTEGRATED_DATA_PATH = RAW_DATA_DIR / "장림펌프장 데이터_통합.csv"
RAINFALL_DATA_PATH = RAW_DATA_DIR / "20251001_사하_강수량.csv"
METADATA_PATH = RAW_DATA_DIR / "20251001_장림유수지_데이터정리_Ver2.xlsx"

# 모델 하이퍼파라미터
SEQUENCE_LENGTH = 60  # 과거 60분 데이터 사용
PREDICTION_HORIZONS = [10, 30, 60]  # 10분, 30분, 60분 후 예측
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
VALIDATION_SPLIT = 0.2

# 데이터 분할 비율
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 타겟 변수
TARGET_COLUMNS = ['SYSTEM.LT_102', 'SYSTEM.LT_103']  # 유수지 A, B
TARGET_NAMES = ['reservoir_A', 'reservoir_B']

# 성능 목표
TARGET_RMSE = 0.05  # 5cm
