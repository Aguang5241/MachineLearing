import numpy as np
import pandas as pd

df = pd.read_csv('PandasStudy/res/data02.csv')
print(df)

# 丢弃
print(df.dropna(axis=0, how='any'))
print(df.dropna(axis=0, how='all'))

# 填充
print(df.fillna(0))

# 判断
print(df.isna())