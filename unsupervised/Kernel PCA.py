import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons

# 비선형 데이터 생성 (두 개의 반달 모양)
X, y = make_moons(n_samples=100, noise=0.05, random_state=42)

# Kernel PCA 적용 (RBF 커널 사용)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X)

# 결과 시각화
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='coolwarm')
plt.xlabel("주성분 1")
plt.ylabel("주성분 2")
plt.title("Kernel PCA (RBF 커널)")
plt.show()
