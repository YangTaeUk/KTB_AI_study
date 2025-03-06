# 필요한 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. 데이터 로드: 붓꽃 데이터셋 사용
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # 꽃잎 길이와 너비만 사용 (2차원 시각화를 위해)
y = iris.target  # 품종 레이블 (0: Setosa, 1: Versicolor, 2: Virginica)

# 데이터 정보 출력
print("데이터 크기:", X.shape)
print("클래스 종류:", np.unique(y))

# 2. 데이터 전처리: 훈련/테스트 데이터 분리 및 정규화
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()  # 데이터 스케일링 (거리 계산의 왜곡 방지)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. k-NN 모델 생성 및 학습
k = 5  # 이웃 수 설정
knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean')
knn.fit(X_train_scaled, y_train)  # 훈련 데이터로 모델 학습

# 4. 테스트 데이터로 예측
y_pred = knn.predict(X_test_scaled)

# 5. 모델 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도 (k={k}): {accuracy:.2f}")


# 6. 시각화: 데이터와 결정 경계
def plot_decision_boundary(X, y, model, title):
    h = 0.02  # 결정 경계 해상도
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 테스트 포인트 생성 및 예측
    test_points = np.c_[xx.ravel(), yy.ravel()]  # 2D 배열로 명확히 변환
    Z = model.predict(test_points)  # 예측 결과 계산
    Z = Z.reshape(xx.shape)  # 그리드 형태로 변환

    # 결정 경계와 데이터 포인트 플롯
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel('꽃잎 길이 (정규화)')
    plt.ylabel('꽃잎 너비 (정규화)')
    plt.title(title)
    plt.legend(handles=scatter.legend_elements()[0], labels=iris.target_names)
    plt.show()


# 훈련 데이터로 결정 경계 시각화
plot_decision_boundary(X_train_scaled, y_train, knn, f'k-NN 분류 (k={k}) - 훈련 데이터')

# 테스트 데이터로 결과 확인
plot_decision_boundary(X_test_scaled, y_test, knn, f'k-NN 분류 (k={k}) - 테스트 데이터')