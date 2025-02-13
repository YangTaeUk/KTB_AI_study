"""
train.py

가상의 선형 데이터를 생성해 회귀 모델을 학습하고,
결과를 model.pkl 파일로 저장하는 스크립트
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import joblib  # 모델 저장/로드를 위한 라이브러리

def train_and_save_model():
    # 1) 가상 데이터 생성
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)  # 0~2 사이 난수
    y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + 잡음

    # 2) 모델 학습
    model = LinearRegression()
    model.fit(X, y)

    # 3) 모델 저장
    joblib.dump(model, "model.pkl")
    print("Model saved to model.pkl")

if __name__ == "__main__":
    train_and_save_model()
