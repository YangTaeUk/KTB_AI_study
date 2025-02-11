import numpy as np
# array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# print(np.empty_like(array))
#
# array = np.array([10, 20, 30, 40, 50, 60]).reshape(2,3)
# print(array)
#
# array = np.array([7, 14, 21])
# array = array[np.newaxis, :]
# print(array.ndim)

#
# array = np.array([[1, 2, 3], [4, 5, 6]])
# array = np.array([10, 20, 30, 40, 50, 60]).reshape(2,3)

# array = np.array([
#     [[1, 2], [3, 4]],
#     [[5, 6], [7, 8]],
#     [[9, 10], [11, 12]]
# ]).reshape(3, 2, 2)
# print(array.shape)

# array = np.array([10, 20, 30, 40, 50])
# print(array[0])
# print(array[-1])

# matrix = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ])
# print(matrix[:,0])
# print(matrix[1])


# array = np.array([5, 15, 8, 20, 3, 12])
# print(np.where(array > 10))

# matrix = np.array([[10, 20, 30], [40, 50, 60]])
# vector = np.array([1, 2, 3])
# print(matrix+vector)

# array = np.array([[3, 7, 2], [8, 4, 6]])
# print(np.max(array, axis=1).shape)

# array1 = np.array([1, 2, 3, 4])
# print(np.power(array1, 2))

# array1 = np.array([10, 20, 30])
# array2 = np.array([1, 2, 3])
# array1+=array2
# print(array1)

# array = np.array([1, np.e, 10, 100])
# arraylog = np.log(array)
# print(arraylog[arraylog > 1])

import pandas as pd

# series = pd.Series([5, 10, 15, 20])
# print(series.index)

# data = {'a': 100, 'b': 200, 'c': 300}
# series = pd.Series(data)
# print(series['b'])


# series = pd.Series([1, 2, None, 4, None, 6])
# print(series.isnull())
# series = series.fillna(0)
# print(series)

# data = {'이름': ['홍길동', '김철수', '박영희', '이순신'],
#         '국어': [85, 90, 88, 92],
#         '영어': [78, 85, 89, 87],
#         '수학': [92, 88, 84, 90]}
# df = pd.DataFrame(data)
# df['총점'] = df.iloc[:, 1:].sum(axis=1)
# over_index = df[df['총점'] > 256]
# print(over_index)

# data = {'이름': ['홍길동', '김철수', '박영희', '이순신', '강감찬'],
#         '나이': [25, 30, 35, 40, 45],
#         '도시': ['서울', '부산', '서울', '대구', '부산']}
#
# df = pd.DataFrame(data)
# over = df[df['나이'] > 30]
# print(over)

# data = {'이름': ['홍길동', '김철수', '박영희', '이순신', '강감찬'],
#         '나이': [25, 30, 35, 40, 45],
#         '도시': ['서울', '부산', '서울', '대구', '부산'],
#         '점수': [85, 90, 75, 95, 80]}
#
# df = pd.DataFrame(data)
# over = df[(df['도시'] == '서울') | (df['점수'] >= 80)]
# print(over)

# data = {'이름': ['홍길동', '김철수', '박영희', '이순신', '강감찬'],
#         '나이': [25, 30, 35, 40, 45],
#         '도시': ['서울', '부산', '서울', '대구', '부산'],
#         '점수': [85, 90, 75, 95, 80]}
#
# df = pd.DataFrame(data)
# filtered_df = df.query("나이 >= 35 and 점수 > 80")
#
# print(filtered_df)

# data = {
#     '이름': ['홍길동', '김철수', '박영희', '이순신'],
#     '부서': ['영업', '영업', '인사', '인사'],
#     '급여': [5000, 5500, 4800, 5100]
# }
#
# df = pd.DataFrame(data)
# print(df.groupby('부서')['급여'].sum())

# data = {
#     '이름': ['홍길동', '김철수', '박영희', '이순신', '강감찬', '신사임당'],
#     '부서': ['영업', '영업', '인사', '인사', 'IT', 'IT'],
#     '급여': [5000, 5500, 4800, 5100, 6000, 6200]
# }
#
# df = pd.DataFrame(data)
# df_g = df.groupby('부서').filter(lambda x: x['급여'].mean() >= 5000)
#
# print(df_g)

#
# df1 = pd.DataFrame({'고객ID': [1, 2, 3], '이름': ['홍길동', '김철수', '이영희']})
# df2 = pd.DataFrame({'고객ID': [2, 3, 4], '구매액': [10000, 20000, 30000]})
#
# print(df1.merge(df2, how='inner', on='고객ID'))


# df1 = pd.DataFrame({'고객ID': [1, 2, 3], '이름': ['홍길동', '김철수', '이영희']})
# df2 = pd.DataFrame({'고객ID': [2, 3, 4], '구매액': [15000, 25000, 35000]})
# print(df1.join(df2, how='left', on='고객ID', lsuffix='_left', rsuffix='_right'))

# df1 = pd.DataFrame({
#     '고객ID': [1, 2, 3],
#     '도시': ['서울', '부산', '대전'],
#     '구매액': [10000, 20000, 30000]
# })
#
# df2 = pd.DataFrame({
#     '고객ID': [1, 2, 3],
#     '도시': ['서울', '부산', '광주'],
#     '구매액': [15000, 25000, 35000]
# })
#
# print(df1.join(df2, on='고객ID', how='outer', lsuffix='_l', rsuffix='_r'))

# data = {'이름': ['홍길동', '김철수', np.nan, '이영희'],
#         '나이': [25, np.nan, 30, 28],
#         '성별': ['남', '남', '여', np.nan]}
#
# df = pd.DataFrame(data)
# print(df.isnull())
# print(df.isnull().sum(axis=0))
#
# print(df.dropna())
#
# print(df['나이'].mean())
# df = df.fillna(df.mean(numeric_only=True))
# #df['나이'] = df['나이'].fillna(df['나이'].mean())
# print(df)

# data = {
#     '날짜': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
#     '제품': ['A', 'B', 'A', 'B'],
#     '판매량': [100, 200, 150, 250]
# }
#
# df = pd.DataFrame(data)
#
# print(df.pivot(index='날짜', columns='제품',values='판매량'))

# data = {
#     '카테고리': ['전자', '가전', '전자', '가전'],
#     '제품': ['A', 'B', 'A', 'B'],
#     '판매량': [100, 200, 150, 250]
# }
#
# df = pd.DataFrame(data)
# print(df.pivot_table(index='카테고리', columns='제품',values='판매량', aggfunc='sum'))

# data = {
#     '날짜': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
#     '제품': ['A', 'B', 'A', 'B'],
#     '판매량': [100, 200, 150, 250],
#     '이익': [20, 50, 30, 60]
# }
#
# df = pd.DataFrame(data)
# print(df.pivot(index='날짜', columns='제품',values=['판매량', '이익']))

# data = {
#     '이름': ['김철수', '이영희', '김철수', '박민수'],
#     '나이': [25, 30, 25, 40],
#     '성별': ['남', '여', '남', '남']
# }
#
# df = pd.DataFrame(data)
# print(df.drop_duplicates())

# data = {
#     '제품': ['노트북', '태블릿', '노트북', '스마트폰'],
#     '가격': [1500000, 800000, 1500000, 1000000],
#     '카테고리': ['전자기기', '전자기기', '전자기기', '전자기기']
# }
#
# df = pd.DataFrame(data)
# df.drop_duplicates(subset=['제품'], inplace=True)
#
# print(df.duplicated().sum())

# data = {
#     '학생': ['김민수', '박지현', '김민수', '이정훈'],
#     '성적': [90, 85, 90, 88],
#     '학교': ['A고', 'B고', 'A고', 'C고']
# }
#
# df = pd.DataFrame(data)
# df_unique = df.drop_duplicates(keep=False)
# df_unique.to_csv('filtered_data.csv', index=False)
# df_loaded = pd.read_csv('filtered_data.csv')
# print(df_loaded)


# data = pd.Series(["HELLO", "WORLD", "PYTHON", "PANDAS"])
# print(data.str.lower())


# df = pd.DataFrame({"이름": [" John Doe ", "Alice ", " Bob ", "Charlie Doe "]})
#
# print(df["이름"].str.strip())
# print(df[df["이름"].str.contains("Doe")])

df = pd.DataFrame({"설명": ["빅데이터 분석", "데이터 과학", "머신 러닝", "딥 러닝"]})
sp = df["설명"].apply(lambda x: "".join([y[0] for y in x.split()]))
df["약어"] = sp
print(df)
df["약어"] = df["설명"].fillna("").apply(lambda x: "".join([y[0] for y in x.split()]))
print(df)