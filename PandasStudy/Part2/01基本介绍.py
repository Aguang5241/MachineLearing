import numpy as np
import pandas as pd

index = pd.date_range('20200219', periods=6)
print(index)

df = pd.DataFrame(np.random.randn(6, 4), index=index, columns=['a', 'b', 'c', 'd'])
print(df)

df2 = pd.DataFrame(np.random.randn(6, 4))
print(df2)

# 概述
print(df2.describe())
# 排序
print(df2.sort_index(ascending=False))
print(df2.sort_values(by=3))