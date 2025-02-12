from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
model = LinearRegression()

np.random.seed(42)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)  # y = 3X + 4 + 잡음


model.fit(x, y)
y_predict = model.predict(x)

plt.scatter(x, y, color="blue", label="Actual Data")
plt.plot(x, y_predict, color="red", label="Linear Regression" )
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

print(f"학습된 가중치 (w): {model.coef_[0]}")
print(f"학습된 절편 (b): {model.intercept_}")