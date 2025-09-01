# C:/githome/17week_cloud-data-viz2/3_와인분류모델_실습.py
# # 와인 데이터 분류를 위한 딥러닝 모델 만들기

# ## 1. 라이브러리 임포트 및 데이터 준비

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 와인 데이터셋 로드 (웹에서 직접)
# 레드 와인과 화이트 와인 데이터가 있으며, type 컬럼으로 구분됩니다.
red_wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
white_wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

red_wine = pd.read_csv(red_wine_url, sep=';')
white_wine = pd.read_csv(white_wine_url, sep=';')

# 데이터 구분을 위해 type 컬럼 추가
red_wine['type'] = 0  # 레드 와인
white_wine['type'] = 1 # 화이트 와인

# 데이터 합치기
wine = pd.concat([red_wine, white_wine])

print('데이터 샘플:')
print(wine.head())
print('\n데이터 정보:')
wine.info()

# ## 2. 데이터 전처리

# Feature(X)와 Target(y) 분리
X = wine.drop('type', axis=1)
y = wine['type']

# 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 데이터 스케일링 (StandardScaler 사용)
# 각 특성의 평균을 0, 분산을 1로 조정하여 모델 성능을 향상시킵니다.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('훈련 데이터 shape:', X_train_scaled.shape)
print('테스트 데이터 shape:', X_test_scaled.shape)

# ## 3. 딥러닝 모델 구성 (Keras)

# Sequential API를 사용하여 모델을 구성합니다.
model = Sequential([
    # 입력층 (Input Layer) - 12개의 특성을 입력으로 받습니다.
    Dense(30, activation='relu', input_shape=(12,)),
    
    # 은닉층 (Hidden Layer)
    Dense(12, activation='relu'),
    
    # 출력층 (Output Layer) - 레드(0) 또는 화이트(1)를 예측하는 이진 분류 문제이므로, 뉴런은 1개, 활성화 함수는 sigmoid를 사용합니다.
    Dense(1, activation='sigmoid')
])

# 모델 구조 확인
model.summary()

# ## 4. 모델 컴파일 및 학습

# 모델의 학습 과정을 설정합니다 (컴파일).
model.compile(optimizer='adam',
              loss='binary_crossentropy', # 이진 분류 문제이므로 binary_crossentropy를 사용합니다.
              metrics=['accuracy'])

# 모델을 학습시킵니다.
# validation_split=0.2: 훈련 데이터 중 20%를 검증 데이터로 사용하여 각 epoch마다 성능을 모니터링합니다.
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# ## 5. 모델 평가

# 테스트 데이터로 모델의 최종 성능을 평가합니다.
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'테스트 손실(Loss): {loss:.4f}')
print(f'테스트 정확도(Accuracy): {accuracy:.4f}')

# ## 6. 학습 과정 시각화 (선택 사항)

import matplotlib.pyplot as plt

def plot_history(history):
    # 정확도 그래프
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

plot_history(history)
