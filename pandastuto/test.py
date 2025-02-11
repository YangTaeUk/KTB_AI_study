import pandas as pd
# data = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])

# df = pd.read_csv('data.csv')  # CSV 파일 읽기
# df = pd.read_excel('data.xlsx')  # 엑셀 파일 읽기
# df = pd.read_json('data.json')  # JSON 파일 읽기
# df = pd.read_sql('SELECT * FROM table', conn)  # SQL 데이터 읽기
# df.to_csv('output.csv', index=False)  # CSV로 저장
# df.to_excel('output.xlsx', index=False)  # 엑셀로 저장
# df.to_json('output.json')  # JSON으로 저장

test = pd.read_csv('train.csv')
print(test.fillna(0))
# print(test.groupby('sex')['age'].mean())
#
# import matplotlib.pyplot as plt
#
# test['age'].hist()  # 나이 히스토그램
# test.plot(x='sex', y='age', kind='bar')  # 막대 그래프
# plt.show()

