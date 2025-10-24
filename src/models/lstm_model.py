"""LSTM 모델 정의"""
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(input_shape: tuple,
                     output_dim: int = 2,
                     lstm_units: list = [128, 64],
                     dropout_rate: float = 0.2,
                     learning_rate: float = 0.001) -> keras.Model:
    """LSTM 모델 구축

    Args:
        input_shape: (sequence_length, n_features)
        output_dim: 출력 차원 (예측 타겟 수)
        lstm_units: LSTM 레이어별 유닛 수
        dropout_rate: Dropout 비율
        learning_rate: 학습률

    Returns:
        컴파일된 Keras 모델
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(lstm_units[0], return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(lstm_units[1]),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_dim)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    return model


def get_callbacks(model_path: str, patience: int = 10) -> list:
    """학습 콜백 함수"""
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
