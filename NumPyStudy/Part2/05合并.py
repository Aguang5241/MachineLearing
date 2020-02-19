import numpy as np

A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

# 横向合并
print('vertical stack\n', np.vstack((A, B, A)))
# 纵向合并
A = A[:, np.newaxis]
B = B[:, np.newaxis]
print('horizontal stack\n', np.hstack((A, B, A)))
# 自由合并(没啥卵用)
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
C = np.array([7, 8, 9])
print(np.concatenate((A, B, C), axis=0))