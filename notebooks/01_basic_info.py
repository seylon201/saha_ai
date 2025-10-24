"""Step 1: 데이터 기본 정보 확인"""
import sys
sys.path.append('/mnt/c/jangrim-lstm-prediction')

from src.data.loader import load_integrated_data, get_data_info
from src.config import INTEGRATED_DATA_PATH

print("=" * 60)
print("Step 1: 통합 센서 데이터 기본 정보 확인")
print("=" * 60)

# 데이터 로드
print("\n데이터 로딩 중...")
df = load_integrated_data(INTEGRATED_DATA_PATH)

# 기본 정보
info = get_data_info(df)

print(f"\n✓ 데이터 로딩 완료!")
print(f"  - 데이터 shape: {info['shape']}")
print(f"  - 메모리 사용량: {info['memory_usage']:.2f} MB")

print(f"\n✓ 시간 범위:")
print(f"  - 시작: {df['시간'].min()}")
print(f"  - 종료: {df['시간'].max()}")
print(f"  - 기간: {(df['시간'].max() - df['시간'].min()).days}일")

print(f"\n✓ 총 컬럼 수: {len(info['columns'])}")
print("\n처음 5개 컬럼:")
for i, col in enumerate(info['columns'][:5], 1):
    print(f"  {i}. {col}")

print("\n저장된 정보:")
print(f"  - 총 컬럼: {len(info['columns'])}개")
print(f"  - 데이터 타입 분포: {df.dtypes.value_counts().to_dict()}")
