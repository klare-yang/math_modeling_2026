import pandas as pd

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


sorted_df = df.dropna().sort_values(by = "petal_length", ascending = 0)
print(sorted_df.head(10))