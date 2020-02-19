import pandas as pd
import numpy as np

###################Series数据类型########################
###################1.由python字典创建########################
a = pd.Series({'a': 10, 'b': 20, 'c': 30})
print(a)
# a    10
# b    20
# c    30
# dtype: int64
###################2.查看数据类型########################
print(type(a))
# <class 'pandas.core.series.Series'>
###################3.由ndarray创建########################
b = pd.Series(np.random.randn(5))
print(b)
# 0    0.520024
# 1    0.600666
# 2   -1.193891
# 3   -1.043480
# 4    1.227984
# dtype: float64
###################DataFrame数据类型########################
###################4.由Series组成的字典构建########################
c = pd.DataFrame({'one': pd.Series([1, 2, 3]), 'two': pd.Series([4, 5, 6])})
print(c)
#    one  two
# 0    1    4
# 1    2    5
# 2    3    6
###################5.由python列表构建########################
d = pd.DataFrame({'one': [1, 2, 3], 'two': [4, 5, 6]})
print(d)
#    one  two
# 0    1    4
# 1    2    5
# 2    3    6
###################6.由带字典的列表构建########################
e = pd.DataFrame([{'one': 1, 'two': 4},
                  {'one': 2, 'two': 5},
                  {'one': 3, 'two': 6}])
print(e)
#    one  two
# 0    1    4
# 1    2    5
# 2    3    6
###################7.由numpy二维数组来构建########################
f = pd.DataFrame(np.random.randint(5, size=(2, 4)))
print(f)
#    0  1  2  3
# 0  4  0  3  0
# 1  2  1  4  1
###################8.DataFrame与Series区别########################
g1 = pd.Series(np.random.randint(5, size=(5, )))
g2 = pd.DataFrame(np.random.randint(5, size=(5, )))
print(g1)
# 0    0
# 1    3
# 2    4
# 3    4
# 4    0
# dtype: int32
print(g2)
#    0
# 0  0
# 1  3
# 2  4
# 3  4
# 4  0