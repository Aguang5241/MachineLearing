import numpy as np

A = np.arange(9).reshape((3, 3))
print('A--->\n', A)
# 迭代行
for row in A:
    print('row\n', row)
# 迭代列
for col in A.T:
    print('col\n', col)
# 遍历元素
for item in A.flat:
    print(item)