import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = 2 * np.random.rand(10000, 1)
y = 4 + 3 * x +np.random.randn(10000, 1)


learning_rate = 0.1
n_iterations = 1000
m = len(x)

w = np.random.randn(1)
b = np.random.randn(1)

for iterations in range(n_iterations):
    y_pred = w * x + b
    error = y_pred - y

    w_gradient = (2/m) * np.sum(error * x)
    b_gradient = (2/m) * np.sum(error)

    w -= learning_rate * w_gradient
    b -= learning_rate * b_gradient

plt.scatter(x, y, color="blue", label="Actual Data")
plt.plot(x,w*x+b, color="red", label="Linear Regression" )
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

print(f"학습된 가중치 (w): {w}")
print(f"학습된 절편 (b): {b}")