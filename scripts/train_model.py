"""LSTM 모델 학습 스크립트"""
import sys
import os
# Windows와 Linux 경로 모두 지원
if os.name == 'nt':  # Windows
    sys.path.append('C:/jangrim-lstm-prediction')
else:  # Linux/WSL
    sys.path.append('/mnt/c/jangrim-lstm-prediction')

import numpy as np
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow 로그 간소화

from src.models.lstm_model import build_lstm_model, get_callbacks
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

print("=" * 60)
print("LSTM 모델 학습")
print("=" * 60)

# Step 1: 데이터 로드
print("\n[Step 1] 전처리 데이터 로딩...")
X_train = np.load(PROCESSED_DATA_DIR / 'X_train_sample.npy')
y_train = np.load(PROCESSED_DATA_DIR / 'y_train_sample.npy')
X_val = np.load(PROCESSED_DATA_DIR / 'X_val_sample.npy')
y_val = np.load(PROCESSED_DATA_DIR / 'y_val_sample.npy')

print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  X_val: {X_val.shape}")
print(f"  y_val: {y_val.shape}")

# Step 2: 모델 구축
print("\n[Step 2] LSTM 모델 구축...")
input_shape = (X_train.shape[1], X_train.shape[2])  # (60, 2)
output_dim = y_train.shape[1]  # 2

model = build_lstm_model(
    input_shape=input_shape,
    output_dim=output_dim,
    lstm_units=[128, 64],
    dropout_rate=0.2,
    learning_rate=0.001
)

print(f"  입력 shape: {input_shape}")
print(f"  출력 차원: {output_dim}")
print(f"\n모델 구조:")
model.summary()

# Step 3: 콜백 설정
print("\n[Step 3] 콜백 설정...")
model_path = str(MODELS_DIR / 'lstm_model_best.keras')
callbacks = get_callbacks(model_path, patience=10)
print(f"  모델 저장 경로: {model_path}")

# Step 4: 모델 학습
print("\n[Step 4] 모델 학습 시작...")
print(f"  에포크: 50")
print(f"  배치 크기: 32")
print("-" * 60)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Step 5: 학습 결과 저장
print("\n[Step 5] 학습 히스토리 저장...")
history_path = MODELS_DIR / 'training_history.npy'
np.save(history_path, history.history)
print(f"  저장 완료: {history_path}")

# Step 6: 최종 결과
print("\n[Step 6] 학습 결과:")
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_train_mae = history.history['mae'][-1]
final_val_mae = history.history['val_mae'][-1]

best_epoch = np.argmin(history.history['val_loss']) + 1
best_val_loss = np.min(history.history['val_loss'])

print(f"  최종 Train Loss: {final_train_loss:.6f}")
print(f"  최종 Val Loss: {final_val_loss:.6f}")
print(f"  최종 Train MAE: {final_train_mae:.6f}")
print(f"  최종 Val MAE: {final_val_mae:.6f}")
print(f"\n  최고 성능 (Epoch {best_epoch}):")
print(f"    Val Loss: {best_val_loss:.6f}")

print("\n" + "=" * 60)
print("학습 완료!")
print("=" * 60)
