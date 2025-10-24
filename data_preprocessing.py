import pandas as pd
import numpy as np
from datetime import datetime

def load_and_preprocess_data():
    """장림펌프장 데이터에서 유수지 A, B 수위 데이터 추출 및 전처리"""
    
    # 데이터 로드
    df = pd.read_csv('/mnt/c/jangrim-lstm-prediction/data/장림펌프장 데이터_통합.csv')
    
    # 시간 컬럼을 datetime으로 변환
    df['시간'] = pd.to_datetime(df['시간'])
    
    # 유수지 A, B 수위 데이터만 추출
    water_level_data = df[['시간', 'SYSTEM.LT_102', 'SYSTEM.LT_103']].copy()
    water_level_data.columns = ['timestamp', 'reservoir_A', 'reservoir_B']
    
    # 결측값 확인
    print("=== 데이터 기본 정보 ===")
    print(f"전체 데이터 수: {len(water_level_data)}")
    print(f"시간 범위: {water_level_data['timestamp'].min()} ~ {water_level_data['timestamp'].max()}")
    print(f"유수지 A 결측값: {water_level_data['reservoir_A'].isna().sum()}")
    print(f"유수지 B 결측값: {water_level_data['reservoir_B'].isna().sum()}")
    
    # 기본 통계
    print("\n=== 기본 통계 ===")
    print(water_level_data.describe())
    
    return water_level_data

def create_sequences(data, sequence_length=60, prediction_length=30):
    """시계열 데이터를 LSTM 입력 형태로 변환"""
    
    # 결측값 처리 (선형 보간)
    data_filled = data.interpolate(method='linear')
    
    # 60분 데이터로 30분 후 예측하는 시퀀스 생성
    X, y = [], []
    
    for i in range(sequence_length, len(data_filled) - prediction_length):
        # 과거 60분 데이터 (reservoir_A, reservoir_B)
        X.append(data_filled[['reservoir_A', 'reservoir_B']].iloc[i-sequence_length:i].values)
        # 30분 후 수위 (reservoir_A, reservoir_B)
        y.append(data_filled[['reservoir_A', 'reservoir_B']].iloc[i+prediction_length].values)
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # 데이터 로드 및 전처리
    data = load_and_preprocess_data()
    
    # 시퀀스 생성
    X, y = create_sequences(data)
    
    print(f"\n=== 시퀀스 생성 결과 ===")
    print(f"입력 데이터 shape: {X.shape}")  # (샘플 수, 60분, 2변수)
    print(f"출력 데이터 shape: {y.shape}")  # (샘플 수, 2변수)
    
    # 샘플 데이터 저장
    np.save('/mnt/c/jangrim-lstm-prediction/X_sequences.npy', X)
    np.save('/mnt/c/jangrim-lstm-prediction/y_sequences.npy', y)
    
    print("\n시퀀스 데이터가 저장되었습니다.")