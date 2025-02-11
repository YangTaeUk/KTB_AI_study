import numpy as np
# 📌 1. 기초 연습문제 (Basic)
# ✅ 배열 생성 및 기초 연산
# arr = np.zeros(10)
# print("길이가 10이고, 모든 원소가 0인 NumPy 배열을 생성하세요.",arr)
# arr = np.ones(10)
# print("길이가 10이고, 모든 원소가 1인 NumPy 배열을 생성하세요.",arr)
# arr = np.full(10, 9)
# print("길이가 10이고, 모든 원소가 9인 NumPy 배열을 생성하세요.",arr)
# arr = np.arange(9)
# print("0부터 9까지의 숫자로 이루어진 배열을 생성하세요.",arr)
# arr = np.where(arr % 2 == 0, 0, arr)
# print("0부터 9까지의 숫자가 들어있는 배열에서 모든 짝수를 0으로 변경하세요.",arr)
# arr = np.arange(9)
# arr2 = arr.reshape(3,3)
# print("1차원 배열을 만들고, 이를 3x3 행렬로 변환하세요.",arr, arr2)
#
# print("주어진 배열에서 최대값, 최소값, 평균을 구하세요.", arr.min(), arr.max(), arr.mean())
# arr = np.random.rand(10,10)
# arr = np.random.randn(10,10)
# arr = np.random.randint(1,100,(10,10))
# print(arr)
# print("1️⃣ 100개의 랜덤 숫자로 이루어진 배열을 만들고 평균을 구하세요.", arr.mean())
# print("2️⃣ 3x3 정수 랜덤 행렬을 만들고, 각 행의 최대값과 최소값을 구하세요.", np.max(arr, axis=1), np.min(arr, axis=1))
# arr = np.eye(5,5)
# print(arr)
# indices = [(i,i) for i in range(1, 4)]
# # (1,1)과 (2,2) 위치 선택 후 값 변경
# # indices = (np.array([1, 2]), np.array([1, 2]))  # 원하는 좌표 설정
# # arr[indices] = 9  # 선택된 위치를 9로 변경
# # print(arr)
# arr[indices] = 9
# print("3️⃣ 5x5 단위행렬을 만들고, (1,1)부터 (3,3)까지 값을 9로 채우세요.",arr)
# arr = np.arange(10)
# print("4️⃣ 1차원 배열을 오름차순,내림차순으로 정렬하세요.", np.sort(arr)[::-1])
# arr = np.arange(1, 10).reshape(3,3)
# print(arr)
# arr2 = np.arange(1, 10).reshape(3,3)
# print(arr2)
# print("5️⃣ 두 개의 2D 배열을 생성하고, 각 원소별로 합을 구하세요.",arr+arr2)

# 1️⃣ np.sort() → 원본 유지, 정렬된 배열 반환
# 2️⃣ .sort() → 원본 배열 직접 정렬 (제자리 정렬)
# 3️⃣ np.argsort() → 정렬된 순서의 인덱스 반환
# 4️⃣ np.lexsort() → 여러 개의 열을 기준으로 정렬
# 5️⃣ axis=0 → 열 기준 정렬, axis=1 → 행 기준 정렬

print("📌 3. 고급 연습문제 (Advanced)")
print("✅ 복잡한 배열 연산")
n = 10
arr = np.random.randint(1,100,(n,n))
print(arr)
print("1부터 100까지의 랜덤 정수로 이루어진 (10x10) 행렬을 생성하고, 각 행의 최대값을 출력하세요.",np.max(arr, axis=1))
arr[n//2,:] = 0
print("10x10 행렬에서 중앙(가운데) 값들을 0으로 변경하세요.", arr)
print("각 열의 평균을 출력하세요.",np.mean(arr, axis=1))
n=6
arr = np.random.randint(1,100,(n,n))
print(arr)
arr[:] = 0
print(arr)
arr[0, :] = 1
arr[:, 0] = 1
arr[n-1, :] = 1
arr[:, n-1] = 1
print("6x6 행렬을 만들고, 가장 바깥쪽 테두리는 1, 나머지는 0으로 채우세요.")
print(arr)
arr = np.random.randint(1,100, 100)
print(arr)
print("100개의 랜덤 숫자로 이루어진 배열을 만들고, 중복된 원소의 개수를 세세요.", np.bincount(arr))
print("주어진 행렬에서 짝수 행만 선택하여 출력하세요.")
print("1차원 배열을 (3x3) 행렬로 변환한 뒤, 전치행렬(Transpose)을 구하세요.")
print("(3x3) 행렬을 만들고, 각 행의 합과 각 열의 합을 출력하세요.")
print("5x5 정규 분포(평균=0, 표준편차=1)를 따르는 난수 행렬을 생성하고, 각 행의 표준편차를 계산하세요.")
print("두 개의 (4x4) 행렬을 생성하고, **행렬 곱(내적)**을 수행하세요.")
print("📌 추가 보너스 문제 (응용)")
print("🔥 실제 데이터 분석 연습")
print("numpy.random을 이용하여 **키(Height)**와 몸무게(Weight) 데이터를 가진 (100x2) 행렬을 생성하고, 평균과 표준편차를 계산하세요.")
print("주어진 데이터에서 상위 10%에 해당하는 값을 찾아 출력하세요.")
print("NumPy를 이용해 이진 탐색(Binary Search) 함수를 구현하세요.")
print("NumPy를 이용해 **가우시안 커널(2D Gaussian Kernel)**을 구현하세요.")
print("numpy.linspace를 사용하여 0~100 사이에서 50개의 균등한 값을 생성하세요.")



