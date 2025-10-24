"""학습 히스토리 분석 스크립트 (matplotlib 없이)"""
import sys
import os
if os.name == 'nt':
    sys.path.append('C:/jangrim-lstm-prediction')
else:
    sys.path.append('/mnt/c/jangrim-lstm-prediction')

import numpy as np
from pathlib import Path

from src.config import MODELS_DIR

print("=" * 60)
print("학습 히스토리 상세 분석")
print("=" * 60)

# Step 1: 학습 히스토리 로드
print("\n[Step 1] 학습 히스토리 로딩...")
history_path = MODELS_DIR / 'training_history.npy'

if not history_path.exists():
    print(f"  ✗ 학습 히스토리 파일이 없습니다: {history_path}")
    print("  학습이 완료될 때까지 기다려주세요.")
    exit(1)

history = np.load(history_path, allow_pickle=True).item()
print(f"  ✓ 로드 완료")
print(f"  총 에포크: {len(history['loss'])}")

# Step 2: 에포크별 상세 정보
print("\n[Step 2] 에포크별 성능 변화")
print("-" * 80)
print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Train MAE':>12} | {'Val MAE':>12}")
print("-" * 80)

for i in range(len(history['loss'])):
    epoch = i + 1
    train_loss = history['loss'][i]
    val_loss = history['val_loss'][i]
    train_mae = history['mae'][i]
    val_mae = history['val_mae'][i]

    # 최고 성능 에포크 표시
    marker = " ⭐" if i == np.argmin(history['val_loss']) else ""

    print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {train_mae:>12.6f} | {val_mae:>12.6f}{marker}")

print("-" * 80)

# Step 3: 주요 통계
print("\n[Step 3] 주요 통계 분석")

best_epoch = np.argmin(history['val_loss']) + 1
best_val_loss = np.min(history['val_loss'])
best_val_mae = history['val_mae'][best_epoch - 1]

final_train_loss = history['loss'][-1]
final_val_loss = history['val_loss'][-1]
final_train_mae = history['mae'][-1]
final_val_mae = history['val_mae'][-1]

print(f"\n  📊 최고 성능 (Epoch {best_epoch}):")
print(f"    Val Loss: {best_val_loss:.6f}")
print(f"    Val MAE: {best_val_mae:.6f} ({best_val_mae*100:.2f} cm)")

print(f"\n  📊 최종 성능 (Epoch {len(history['loss'])}):")
print(f"    Train Loss: {final_train_loss:.6f}")
print(f"    Val Loss: {final_val_loss:.6f}")
print(f"    Train MAE: {final_train_mae:.6f} ({final_train_mae*100:.2f} cm)")
print(f"    Val MAE: {final_val_mae:.6f} ({final_val_mae*100:.2f} cm)")

# Step 4: 과적합 분석
print("\n[Step 4] 과적합 분석")

overfitting_gap = final_val_loss - final_train_loss
overfitting_percent = (overfitting_gap / final_train_loss) * 100 if final_train_loss > 0 else 0

print(f"\n  Loss 차이:")
print(f"    Val Loss - Train Loss = {overfitting_gap:.6f}")
print(f"    상대 비율: {overfitting_percent:.1f}%")

if overfitting_gap > 0.01:
    print(f"  ⚠️  심각한 과적합 발생! (차이: {overfitting_gap:.6f})")
    print(f"  권장사항:")
    print(f"    - Dropout 비율 증가 (현재 0.2 → 0.3~0.4)")
    print(f"    - L2 정규화 추가")
    print(f"    - 데이터 증강")
elif overfitting_gap > 0.001:
    print(f"  ⚠️  경미한 과적합 (차이: {overfitting_gap:.6f})")
    print(f"  모니터링 필요")
else:
    print(f"  ✓ 과적합 없음 (차이: {overfitting_gap:.6f})")

# Step 5: 학습 안정성 분석
print("\n[Step 5] 학습 안정성 분석")

# Val Loss 변화율
val_loss_changes = []
for i in range(1, len(history['val_loss'])):
    change = history['val_loss'][i] - history['val_loss'][i-1]
    val_loss_changes.append(change)

val_loss_std = np.std(val_loss_changes)
val_loss_mean_change = np.mean(val_loss_changes)

print(f"\n  Validation Loss 변화:")
print(f"    평균 변화: {val_loss_mean_change:.6f}")
print(f"    표준편차: {val_loss_std:.6f}")

# 개선 횟수 vs 악화 횟수
improvements = sum(1 for c in val_loss_changes if c < 0)
deteriorations = sum(1 for c in val_loss_changes if c > 0)

print(f"\n  성능 변화 추이:")
print(f"    개선: {improvements}회 ({improvements/len(val_loss_changes)*100:.1f}%)")
print(f"    악화: {deteriorations}회 ({deteriorations/len(val_loss_changes)*100:.1f}%)")

if deteriorations > improvements * 1.5:
    print(f"  ⚠️  불안정한 학습 - 학습률 감소 필요")
elif improvements > deteriorations:
    print(f"  ✓ 안정적인 개선 추세")

# Step 6: 목표 성능 비교
print("\n[Step 6] 목표 성능 대비 분석")

target_mae_cm = 3.0  # 목표: 3cm
current_mae_cm = best_val_mae * 100

print(f"\n  목표: MAE < {target_mae_cm} cm")
print(f"  현재: MAE = {current_mae_cm:.2f} cm")
print(f"  차이: {current_mae_cm - target_mae_cm:.2f} cm")

if current_mae_cm < target_mae_cm:
    print(f"  🎉 목표 달성!")
    performance_ratio = (1 - (current_mae_cm / target_mae_cm)) * 100
    print(f"  목표 대비 {performance_ratio:.1f}% 더 좋은 성능")
else:
    gap_percent = ((current_mae_cm / target_mae_cm) - 1) * 100
    print(f"  ⚠️  목표 미달: {gap_percent:.1f}% 더 개선 필요")
    print(f"\n  개선 방안:")
    print(f"    1. 더 많은 특성 추가 (외수위, 펌프 상태 등)")
    print(f"    2. 시퀀스 길이 조정 (현재 60분 → 90분 or 120분)")
    print(f"    3. 모델 구조 개선 (레이어 추가, LSTM units 증가)")
    print(f"    4. 전체 데이터셋 사용 (현재 샘플 데이터)")

# Step 7: 조기 종료 분석
print("\n[Step 7] 조기 종료 분석")

max_epochs = 50
total_epochs = len(history['loss'])

if total_epochs < max_epochs:
    print(f"  ✓ 조기 종료 발동: {total_epochs}/{max_epochs} 에포크")
    print(f"  이유: Validation Loss가 {10} 에포크 동안 개선 없음")

    # 마지막 개선 에포크 찾기
    epochs_since_best = total_epochs - best_epoch
    print(f"  최고 성능 이후: {epochs_since_best} 에포크")
else:
    print(f"  전체 에포크 완료: {total_epochs}/{max_epochs}")
    print(f"  조기 종료 미발동 - 추가 학습 가능성")

print("\n" + "=" * 60)
print("분석 완료!")
print("=" * 60)
print(f"\n다음 단계:")
print(f"  1. 모델 평가: python3 scripts/evaluate_model.py")
print(f"  2. 결과 시각화: python3 scripts/visualize_results.py")
print(f"  3. 성능 개선 작업 시작")
