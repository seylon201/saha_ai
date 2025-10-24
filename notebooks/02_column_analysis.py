"""Step 2: 컬럼 구조 상세 분석"""
import sys
sys.path.append('/mnt/c/jangrim-lstm-prediction')

from src.data.loader import load_integrated_data
from src.config import INTEGRATED_DATA_PATH, TARGET_COLUMNS
import pandas as pd

print("=" * 60)
print("Step 2: 데이터 컬럼 구조 및 타입 분석")
print("=" * 60)

df = load_integrated_data(INTEGRATED_DATA_PATH)

# 타겟 컬럼 확인
print(f"\n✓ 타겟 변수 확인:")
for col in TARGET_COLUMNS:
    if col in df.columns:
        print(f"  ✓ {col} - 존재함")
    else:
        print(f"  ✗ {col} - 없음!")

# 컬럼명 패턴 분석
print(f"\n✓ 컬럼명 패턴 분석:")
patterns = {}
for col in df.columns:
    if col == '시간':
        continue
    prefix = col.split('.')[0] if '.' in col else 'OTHER'
    patterns[prefix] = patterns.get(prefix, 0) + 1

for prefix, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {prefix}: {count}개")

# 수위 관련 컬럼 찾기
print(f"\n✓ 수위(LT) 관련 컬럼:")
lt_columns = [col for col in df.columns if 'LT' in col.upper()]
print(f"  총 {len(lt_columns)}개 발견")
for i, col in enumerate(lt_columns[:10], 1):
    print(f"  {i}. {col}")
if len(lt_columns) > 10:
    print(f"  ... 외 {len(lt_columns)-10}개")

# 펌프 관련 컬럼
print(f"\n✓ 펌프(P) 관련 컬럼:")
pump_columns = [col for col in df.columns if 'P_' in col or '.P' in col]
print(f"  총 {len(pump_columns)}개")
