# 필요한 라이브러리 불러오기
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. 데이터 로드
iris = load_iris()
X = iris.data      # 특성 (예: 꽃받침 길이, 꽃받침 너비 등)
y = iris.target    # 클래스 레이블 (세 종류의 붓꽃)

# 2. 데이터 분할: 학습용과 테스트용 데이터로 분리 (예: 70% 학습, 30% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. k-최근접 이웃 분류기 생성
# 여기서 k=3으로 설정 (즉, 가장 가까운 3개의 이웃을 참고하여 분류)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# 4. 모델 학습 (k-NN은 실제로 데이터를 저장하는 lazy learning 방식)
knn.fit(X_train, y_train)

# 5. 테스트 데이터에 대한 예측 수행
y_pred = knn.predict(X_test)

# 6. 모델 평가: 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# 추가: 예측 결과와 실제 레이블 비교 출력
print("Predicted labels:", y_pred)
print("Actual labels:   ", y_test)

# 선택사항: 간단한 산점도를 통해 일부 특성 시각화
plt.figure(figsize=(8, 6))
# 첫 번째와 두 번째 특성을 사용하여 시각화 (꽃받침 길이와 너비)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='o', label='Predicted')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('k-NN Classification on Iris Test Data')
plt.legend()
plt.show()
