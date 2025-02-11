import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
np.random.seed(42) # 42차원 데이터 랜덤 생성
X = np.random.randn(100,2) # @ np.array([[2,1],[1,1]])

# 변환 행렬
transform_matrix = np.array([[2, 1], [1, 1]])

# 변환 적용
X_transformed = X @ transform_matrix

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transformed)

pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

print("주성분 벡터:\n", pca.components_) # ??
print("설명된 분산 비율:", pca.explained_variance_ratio_)


plt.scatter(X_scaled[:, 0], X_scaled[:, 1], label="Original Data")
plt.scatter(X_pca, np.zeros_like(X_pca), label="PCA Projection", color="red")
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("PCA 차원 축소 예제")
plt.show()

# 원본 데이터 시각화
plt.figure(figsize=(8,4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], color='blue', alpha=0.5, label="Original Data")
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()

# 변환된 데이터 시각화
plt.subplot(1, 2, 2)
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], color='red', alpha=0.5, label="Transformed Data")
plt.title("Transformed Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()

plt.show()