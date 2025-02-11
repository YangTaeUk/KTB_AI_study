import numpy as np

# 샘플 데이터 (각 열이 하나의 변수, 각 행이 한 개의 데이터 포인트)
data = np.array([[2.1, 2.5, 3.6],
                 [1.8, 2.3, 3.2],
                 [2.5, 2.8, 3.8],
                 [2.3, 2.6, 3.7]])

# 공분산 행렬 계산
cov_matrix = np.cov(data, rowvar=False)

print("공분산 행렬:\n", cov_matrix)
