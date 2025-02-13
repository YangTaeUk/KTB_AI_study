import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
x = 2 * np.random.rand(100, 1) # 0~2 사이 균일 분포
y = 4 + 3 * x + np.random.randn(100, 1) # 선형 식 y = 4 + 3 * x

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

print("선형회귀 기울기(β1) : ", lin_reg.coef_)
print("선형회귀 절편(β0) : ", lin_reg.coef_)


