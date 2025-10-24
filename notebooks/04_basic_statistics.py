"""Step 4: 기초 통계 분석"""
import sys
sys.path.append('/mnt/c/jangrim-lstm-prediction')

from src.data.loader import load_integrated_data
from src.config import INTEGRATED_DATA_PATH, TARGET_COLUMNS

print("=" * 60)
print("Step 4: 기초 통계 분석")
print("=" * 60)

df = load_integrated_data(INTEGRATED_DATA_PATH)

# 타겟 변수 기초 통계
print(f"\n✓ 타겟 변수(유수지 A, B) 기초 통계:")
target_stats = df[TARGET_COLUMNS].describe()
print(target_stats)

# 수위 범위
print(f"\n✓ 유수지 수위 범위:")
for col in TARGET_COLUMNS:
    min_val = df[col].min()
    max_val = df[col].max()
    mean_val = df[col].mean()
    std_val = df[col].std()
    print(f"\n  {col}:")
    print(f"    - 최소: {min_val:.2f}m")
    print(f"    - 최대: {max_val:.2f}m")
    print(f"    - 평균: {mean_val:.2f}m")
    print(f"    - 표준편차: {std_val:.2f}m")
    print(f"    - 변동폭: {max_val - min_val:.2f}m")

# 주요 수위 센서들
print(f"\n✓ 주요 수위 센서 요약:")
lt_cols = ['SYSTEM.LT_101', 'SYSTEM.LT_102', 'SYSTEM.LT_103', 'SYSTEM.LT_104']
for col in lt_cols:
    if col in df.columns:
        print(f"  {col}: {df[col].min():.2f}m ~ {df[col].max():.2f}m (평균: {df[col].mean():.2f}m)")
