import numpy as np

a = np.array([1 ,2, 3])

b = a
c = a
d = b
copy = a.copy()

a[0] = 0

print(a)
# [0 2 3]
print(b)
# [0 2 3]
print(c)
# [0 2 3]
print(d)
# [0 2 3]
print(copy)
# [1 2 3]