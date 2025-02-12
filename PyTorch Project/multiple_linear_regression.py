from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
model = LinearRegression()

np.random.seed(42)
x_multi = np.random.rand(100, 3)
y_multi = 3 * x_multi[:, 0] + 7 * x_multi[:, 1] + 12 * x_multi[:, 2] + 4 + np.random.randn(100)  # y = 3X + 4 + 잡음

model.fit(x_multi, y_multi)
y_predict = model.predict(x_multi)

print(f"학습된 가중치 (w): {model.coef_}")
print(f"학습된 절편 (b): {model.intercept_}")