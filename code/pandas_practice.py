import pandas as pd
import numpy as np

df = pd.read_excel(io = "data/IRIS.xlsx", sheet_name = "IRIS", engine = "openpyxl")

# # 基础信息
# print(df.head(10))
# print(df.info())

# # 缺失值处理
# df = df.dropna()
# df['sepal_length'] = df['sepal_length'].astype(int)
# print(df.info())
# print(df.head(10))

# # 选择和过滤
# print(df['sepal_length'] == 4.7)
# print(df['sepal_width'].std())
# lb = df['sepal_width'].mean() - 3 * df['sepal_width'].std()
# ub = df['sepal_width'].mean() + 3 * df['sepal_width'].std()
# new_df = df[(df['sepal_width'] > lb) & (df['sepal_width'] < ub)]
# print(df.info())
# print(new_df.info())

# clean_df = df.dropna()
# sorted_df = df.sort_values(by = "petal_length", ascending = 0)
# print(clean_df.info())
# print(sorted_df.dropna().info())
# print(sorted_df.dropna().head(10))

data = {'name': ['Jack', 'John', 'Peter', 'David', 'Daniel'],
        'hight': [175 for i in range(5)],
        'weight': [65 for i in range(5)],
        'scores': np.random.randint(40,90,5)}
df = pd.DataFrame(data)
print(df)