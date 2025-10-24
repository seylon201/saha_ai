"""모델 성능 분석 리포트 생성"""
import sys
import os
if os.name == 'nt':
    sys.path.append('C:/jangrim-lstm-prediction')
else:
    sys.path.append('/mnt/c/jangrim-lstm-prediction')

import numpy as np
from pathlib import Path
from datetime import datetime

from src.config import MODELS_DIR, RESULTS_DIR

print("=" * 80)
print("모델 성능 분석 리포트 생성")
print("=" * 80)

# 학습 히스토리 로드
history_path = MODELS_DIR / 'training_history.npy'
history = np.load(history_path, allow_pickle=True).item()

# 평가 결과 로드 (있으면)
eval_path = MODELS_DIR / 'evaluation_results.npy'
has_eval = eval_path.exists()

# 리포트 생성
report = []
report.append("# 🎯 LSTM 수위 예측 모델 - 성능 분석 리포트\n")
report.append(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report.append(f"**프로젝트**: 장림 유수지 LSTM 수위 예측\n")
report.append("\n---\n")

# 1. 학습 요약
report.append("\n## 📊 1. 학습 요약\n")
total_epochs = len(history['loss'])
best_epoch = np.argmin(history['val_loss']) + 1
best_val_loss = np.min(history['val_loss'])
best_val_mae = history['val_mae'][best_epoch - 1]

report.append(f"\n### 학습 설정\n")
report.append(f"- **총 에포크**: {total_epochs} / 50 (조기 종료)\n")
report.append(f"- **배치 크기**: 32\n")
report.append(f"- **학습률**: 0.001 (Adam)\n")
report.append(f"- **시퀀스 길이**: 60분\n")
report.append(f"- **예측 시점**: 30분 후\n")

report.append(f"\n### 모델 구조\n")
report.append(f"```\n")
report.append(f"입력: (60, 2) - 60분간 유수지 A, B 수위\n")
report.append(f"LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2)\n")
report.append(f"Dense(32) → Dense(2) - 유수지 A, B 예측\n")
report.append(f"총 파라미터: 118,626개\n")
report.append(f"```\n")

# 2. 최고 성능
report.append(f"\n## 🏆 2. 최고 성능 (Epoch {best_epoch})\n")
report.append(f"\n| 지표 | 값 | 단위 |\n")
report.append(f"|------|------|------|\n")
report.append(f"| Validation Loss | {best_val_loss:.6f} | MSE |\n")
report.append(f"| Validation MAE | {best_val_mae:.6f} | m |\n")
report.append(f"| **Validation MAE** | **{best_val_mae*100:.2f}** | **cm** |\n")

# 3. 최종 성능
final_train_loss = history['loss'][-1]
final_val_loss = history['val_loss'][-1]
final_train_mae = history['mae'][-1]
final_val_mae = history['val_mae'][-1]

report.append(f"\n## 📈 3. 최종 성능 (Epoch {total_epochs})\n")
report.append(f"\n| 구분 | Loss (MSE) | MAE (m) | MAE (cm) |\n")
report.append(f"|------|------------|---------|----------|\n")
report.append(f"| Train | {final_train_loss:.6f} | {final_train_mae:.6f} | {final_train_mae*100:.2f} |\n")
report.append(f"| Validation | {final_val_loss:.6f} | {final_val_mae:.6f} | {final_val_mae*100:.2f} |\n")
report.append(f"| **차이** | **{final_val_loss - final_train_loss:.6f}** | **{final_val_mae - final_train_mae:.6f}** | **{(final_val_mae - final_train_mae)*100:.2f}** |\n")

# 4. 목표 대비 성능
target_mae_cm = 3.0
current_mae_cm = best_val_mae * 100
gap_cm = current_mae_cm - target_mae_cm
gap_percent = (gap_cm / target_mae_cm) * 100

report.append(f"\n## 🎯 4. 목표 대비 성능\n")
report.append(f"\n| 구분 | MAE (cm) | 상태 |\n")
report.append(f"|------|----------|------|\n")
report.append(f"| 목표 | {target_mae_cm:.1f} | - |\n")
report.append(f"| 현재 (최고) | {current_mae_cm:.2f} | {'✅ 달성' if gap_cm <= 0 else '⚠️ 미달'} |\n")
report.append(f"| 차이 | {gap_cm:.2f} | {gap_percent:+.1f}% |\n")

# 5. 문제점 분석
report.append(f"\n## ⚠️ 5. 문제점 분석\n")

overfitting_gap = final_val_loss - final_train_loss
report.append(f"\n### 5.1 과적합 (Overfitting)\n")
report.append(f"- **Loss 차이**: {overfitting_gap:.6f}\n")
report.append(f"- **심각도**: {'🔴 심각' if overfitting_gap > 0.01 else '🟡 경미' if overfitting_gap > 0.001 else '🟢 정상'}\n")

if overfitting_gap > 0.01:
    report.append(f"\n**증상**:\n")
    report.append(f"- Train Loss는 거의 0에 수렴 ({final_train_loss:.6f})\n")
    report.append(f"- Validation Loss는 높게 유지 ({final_val_loss:.6f})\n")
    report.append(f"- 모델이 훈련 데이터를 암기하고 일반화 실패\n")

report.append(f"\n### 5.2 성능 미달\n")
report.append(f"- **목표**: MAE < 3.0 cm\n")
report.append(f"- **현재**: MAE = {current_mae_cm:.2f} cm\n")
report.append(f"- **부족**: {gap_percent:.1f}% 더 개선 필요\n")

# 6. 개선 방안
report.append(f"\n## 💡 6. 개선 방안\n")

report.append(f"\n### 6.1 과적합 해결 (우선순위 1)\n")
report.append(f"\n**A. Dropout 증가**\n")
report.append(f"```python\n")
report.append(f"# 현재: Dropout(0.2)\n")
report.append(f"# 변경: Dropout(0.3) 또는 Dropout(0.4)\n")
report.append(f"```\n")

report.append(f"\n**B. L2 정규화 추가**\n")
report.append(f"```python\n")
report.append(f"from tensorflow.keras import regularizers\n")
report.append(f"LSTM(128, kernel_regularizer=regularizers.l2(0.001))\n")
report.append(f"```\n")

report.append(f"\n**C. 학습률 감소**\n")
report.append(f"```python\n")
report.append(f"# 현재: learning_rate=0.001\n")
report.append(f"# 변경: learning_rate=0.0001 또는 0.0005\n")
report.append(f"```\n")

report.append(f"\n### 6.2 성능 향상 (우선순위 2)\n")

report.append(f"\n**A. 전체 데이터 사용**\n")
report.append(f"- 현재: 샘플 데이터 (~4만 샘플)\n")
report.append(f"- 변경: 전체 데이터 (~100만 샘플)\n")
report.append(f"- 예상 효과: MAE 20-30% 개선\n")

report.append(f"\n**B. 다변량 입력 추가 (Phase 2)**\n")
report.append(f"- 외수위 (SYSTEM.LT_104)\n")
report.append(f"- 펌프 운영 상태 (게이트 펌프 1~6)\n")
report.append(f"- 변화율 특성 (1분, 5분, 10분)\n")
report.append(f"- 시간 특성 (시각, 요일)\n")

report.append(f"\n**C. 시퀀스 길이 조정**\n")
report.append(f"```python\n")
report.append(f"# 현재: 60분 (1시간)\n")
report.append(f"# 실험: 90분, 120분, 180분\n")
report.append(f"# 더 긴 패턴 학습 가능\n")
report.append(f"```\n")

report.append(f"\n**D. 모델 구조 개선**\n")
report.append(f"```python\n")
report.append(f"# 옵션 1: LSTM units 증가\n")
report.append(f"LSTM(256) → LSTM(128) → LSTM(64)\n")
report.append(f"\n")
report.append(f"# 옵션 2: Bidirectional LSTM\n")
report.append(f"Bidirectional(LSTM(128))\n")
report.append(f"\n")
report.append(f"# 옵션 3: Attention 메커니즘 추가\n")
report.append(f"```\n")

# 7. 다음 단계
report.append(f"\n## 🚀 7. 다음 단계 (권장 순서)\n")

report.append(f"\n### Step 1: 과적합 해결 실험\n")
report.append(f"1. Dropout 0.3으로 재학습\n")
report.append(f"2. L2 정규화 추가 후 재학습\n")
report.append(f"3. 학습률 0.0005로 재학습\n")
report.append(f"4. 최적 조합 찾기\n")

report.append(f"\n### Step 2: 전체 데이터 학습\n")
report.append(f"1. 전체 데이터 전처리 (100만+ 샘플)\n")
report.append(f"2. 최적 하이퍼파라미터로 학습\n")
report.append(f"3. 성능 평가\n")

report.append(f"\n### Step 3: Phase 2 진입\n")
report.append(f"1. 다변량 특성 추가\n")
report.append(f"2. 모델 재학습\n")
report.append(f"3. 목표 성능 달성 확인\n")

# 8. 에포크별 상세 데이터
report.append(f"\n## 📋 8. 에포크별 상세 데이터\n")
report.append(f"\n| Epoch | Train Loss | Val Loss | Train MAE | Val MAE (cm) |\n")
report.append(f"|-------|------------|----------|-----------|-------------|\n")

for i in range(len(history['loss'])):
    epoch = i + 1
    marker = " ⭐" if i == best_epoch - 1 else ""
    report.append(f"| {epoch}{marker} | {history['loss'][i]:.6f} | {history['val_loss'][i]:.6f} | {history['mae'][i]:.6f} | {history['val_mae'][i]*100:.2f} |\n")

# 평가 결과 추가 (있으면)
if has_eval:
    eval_results = np.load(eval_path, allow_pickle=True).item()

    report.append(f"\n## 🧪 9. 테스트 세트 평가\n")

    if 'metrics' in eval_results:
        metrics = eval_results['metrics']
        report.append(f"\n### 전체 성능\n")
        report.append(f"- RMSE: {metrics['rmse']:.4f} m ({metrics['rmse']*100:.2f} cm)\n")
        report.append(f"- MAE: {metrics['mae']:.4f} m ({metrics['mae']*100:.2f} cm)\n")
        report.append(f"- MAPE: {metrics['mape']:.2f}%\n")

        if 'reservoir_a' in metrics and 'reservoir_b' in metrics:
            report.append(f"\n### 유수지별 성능\n")
            report.append(f"\n**유수지 A**:\n")
            report.append(f"- RMSE: {metrics['reservoir_a']['rmse']:.4f} m ({metrics['reservoir_a']['rmse']*100:.2f} cm)\n")
            report.append(f"- MAE: {metrics['reservoir_a']['mae']:.4f} m ({metrics['reservoir_a']['mae']*100:.2f} cm)\n")

            report.append(f"\n**유수지 B**:\n")
            report.append(f"- RMSE: {metrics['reservoir_b']['rmse']:.4f} m ({metrics['reservoir_b']['rmse']*100:.2f} cm)\n")
            report.append(f"- MAE: {metrics['reservoir_b']['mae']:.4f} m ({metrics['reservoir_b']['mae']*100:.2f} cm)\n")

# 리포트 저장
report_text = ''.join(report)
report_path = RESULTS_DIR / 'model_analysis_report.md'
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n✓ 리포트 생성 완료!")
print(f"  저장 위치: {report_path}")

# 콘솔 출력
print("\n" + "=" * 80)
print("주요 결과 요약")
print("=" * 80)
print(f"\n🏆 최고 성능 (Epoch {best_epoch})")
print(f"   Val MAE: {best_val_mae*100:.2f} cm")
print(f"\n🎯 목표 대비")
print(f"   목표: {target_mae_cm:.1f} cm")
print(f"   현재: {current_mae_cm:.2f} cm")
print(f"   차이: {gap_cm:+.2f} cm ({gap_percent:+.1f}%)")
print(f"\n⚠️  주요 문제점")
print(f"   1. 심각한 과적합 (Loss 차이: {overfitting_gap:.6f})")
print(f"   2. 성능 미달 ({gap_percent:.1f}% 개선 필요)")
print(f"\n💡 우선 개선 사항")
print(f"   1. Dropout 증가 (0.2 → 0.3~0.4)")
print(f"   2. L2 정규화 추가")
print(f"   3. 전체 데이터 사용")
print("\n" + "=" * 80)
