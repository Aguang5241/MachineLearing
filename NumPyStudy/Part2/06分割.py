import numpy as np

A = np.arange(12).reshape((3, 4))
print('A--->\n', A)

# 等分
print(np.split(A, 3, axis=0))
# 不等分
print(np.array_split(A, 3, axis=1))
# 横向分
print(np.hsplit(A, 2))
# 纵向分
print(np.vsplit(A, 3))