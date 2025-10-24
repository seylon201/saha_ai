"""Step 3: 결측치 분석"""
import sys
sys.path.append('/mnt/c/jangrim-lstm-prediction')

from src.data.loader import load_integrated_data
from src.config import INTEGRATED_DATA_PATH, TARGET_COLUMNS

print("=" * 60)
print("Step 3: 결측치 분석")
print("=" * 60)

df = load_integrated_data(INTEGRATED_DATA_PATH)

# 전체 결측치 현황
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

print(f"\n✓ 전체 결측치 현황:")
print(f"  - 결측치 있는 컬럼: {(missing > 0).sum()}개 / {len(df.columns)}개")
print(f"  - 완전한 컬럼: {(missing == 0).sum()}개")

# 결측치 많은 컬럼 Top 10
print(f"\n✓ 결측치 많은 컬럼 Top 10:")
top_missing = missing_pct[missing_pct > 0].sort_values(ascending=False).head(10)
for col, pct in top_missing.items():
    print(f"  - {col}: {pct:.2f}%")

# 타겟 변수 결측치
print(f"\n✓ 타겟 변수(유수지 A, B) 결측치:")
for col in TARGET_COLUMNS:
    miss_count = missing[col]
    miss_pct = missing_pct[col]
    print(f"  - {col}: {miss_count}개 ({miss_pct:.4f}%)")

# 수위 센서 결측치
print(f"\n✓ 주요 수위 센서 결측치:")
lt_cols = [col for col in df.columns if col.startswith('SYSTEM.LT_')]
for col in lt_cols:
    miss_count = missing[col]
    miss_pct = missing_pct[col]
    if miss_pct > 0:
        print(f"  - {col}: {miss_count}개 ({miss_pct:.2f}%)")
