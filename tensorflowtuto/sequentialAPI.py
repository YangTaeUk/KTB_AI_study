import tensorflow as tf
import keras
layers = keras.layers
# 모델 생성
model = keras.Sequential([
    layers.Dense(64, activation='relu'),  # 입력층
    layers.Dense(32, activation='relu'),  # 은닉층
    layers.Dense(1, activation='sigmoid')  # 출력층
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 구조 확인
model.summary()
