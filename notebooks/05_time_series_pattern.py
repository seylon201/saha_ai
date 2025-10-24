"""Step 5: 시계열 패턴 확인 (샘플링)"""
import sys
sys.path.append('/mnt/c/jangrim-lstm-prediction')

from src.data.loader import load_integrated_data
from src.config import INTEGRATED_DATA_PATH, TARGET_COLUMNS
import matplotlib
matplotlib.use('Agg')  # GUI 없이 저장만
import matplotlib.pyplot as plt

print("=" * 60)
print("Step 5: 타겟 변수 시계열 패턴 확인")
print("=" * 60)

df = load_integrated_data(INTEGRATED_DATA_PATH)

# 최근 7일 데이터만 샘플링 (시각화용)
print("\n✓ 최근 7일 데이터 샘플링...")
df_recent = df.tail(7 * 24 * 60)  # 7일 * 24시간 * 60분

print(f"  샘플 기간: {df_recent['시간'].min()} ~ {df_recent['시간'].max()}")
print(f"  샘플 크기: {len(df_recent):,}개")

# 시계열 그래프 생성
print("\n✓ 시계열 그래프 생성 중...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# 유수지 A
axes[0].plot(df_recent['시간'], df_recent['SYSTEM.LT_102'], linewidth=0.5)
axes[0].set_title('유수지 A 수위 (SYSTEM.LT_102) - 최근 7일')
axes[0].set_ylabel('수위 (m)')
axes[0].grid(True, alpha=0.3)

# 유수지 B
axes[1].plot(df_recent['시간'], df_recent['SYSTEM.LT_103'], linewidth=0.5, color='orange')
axes[1].set_title('유수지 B 수위 (SYSTEM.LT_103) - 최근 7일')
axes[1].set_xlabel('시간')
axes[1].set_ylabel('수위 (m)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = '/mnt/c/jangrim-lstm-prediction/results/figures/01_timeseries_pattern.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ 그래프 저장 완료: {output_path}")

# 변화율 분석
print(f"\n✓ 수위 변화율 분석:")
for col in TARGET_COLUMNS:
    df_recent[f'{col}_diff'] = df_recent[col].diff()
    max_increase = df_recent[f'{col}_diff'].max()
    max_decrease = df_recent[f'{col}_diff'].min()
    print(f"\n  {col}:")
    print(f"    - 최대 상승률: {max_increase:.4f} m/min")
    print(f"    - 최대 하강률: {max_decrease:.4f} m/min")
