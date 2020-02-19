import numpy as np

one = np.array([1, 2, 3])
two = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
three = np.array([[[1, 1], [1, 1], [1, 1]],
                [[1, 1], [1, 1], [1, 1]],
                [[1, 1], [1, 1], [1, 1]]])

print(one.shape)
# (3,)
print(two.shape)
# (3, 3)
print(three.shape)
# (3, 3, 2)