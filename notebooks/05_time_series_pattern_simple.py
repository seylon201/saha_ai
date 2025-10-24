"""Step 5: 시계열 패턴 확인 (텍스트 분석)"""
import sys
sys.path.append('/mnt/c/jangrim-lstm-prediction')

from src.data.loader import load_integrated_data
from src.config import INTEGRATED_DATA_PATH, TARGET_COLUMNS

print("=" * 60)
print("Step 5: 타겟 변수 시계열 패턴 확인")
print("=" * 60)

df = load_integrated_data(INTEGRATED_DATA_PATH)

# 최근 7일 데이터만 샘플링
print("\n✓ 최근 7일 데이터 샘플링...")
df_recent = df.tail(7 * 24 * 60)  # 7일 * 24시간 * 60분

print(f"  샘플 기간: {df_recent['시간'].min()} ~ {df_recent['시간'].max()}")
print(f"  샘플 크기: {len(df_recent):,}개")

# 변화율 분석
print(f"\n✓ 수위 변화율 분석 (최근 7일):")
for col in TARGET_COLUMNS:
    df_recent_copy = df_recent.copy()
    df_recent_copy[f'{col}_diff'] = df_recent_copy[col].diff()

    max_increase = df_recent_copy[f'{col}_diff'].max()
    max_decrease = df_recent_copy[f'{col}_diff'].min()
    mean_change = df_recent_copy[f'{col}_diff'].abs().mean()

    print(f"\n  {col}:")
    print(f"    - 최대 상승률: {max_increase:.4f} m/min")
    print(f"    - 최대 하강률: {max_decrease:.4f} m/min")
    print(f"    - 평균 변화율: {mean_change:.4f} m/min")

# 데이터 샘플링 간격 확인
print(f"\n✓ 데이터 샘플링 간격 확인:")
time_diff = df_recent['시간'].diff().dropna()
print(f"  - 평균 간격: {time_diff.mean()}")
print(f"  - 최소 간격: {time_diff.min()}")
print(f"  - 최대 간격: {time_diff.max()}")

# 전체 기간 요약
print(f"\n✓ 전체 기간 데이터 요약:")
print(f"  - 전체 데이터: {len(df):,}개")
print(f"  - 기간: {(df['시간'].max() - df['시간'].min()).days}일")
print(f"  - 시작: {df['시간'].min()}")
print(f"  - 종료: {df['시간'].max()}")
