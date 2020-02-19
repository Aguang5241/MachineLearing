import numpy as np

# 空
empty = np.empty((2, 2), dtype=np.float64, order='C')
# 0
zeros = np.zeros((2, 2))
# 1
ones = np.ones((2, 2))
# arange(start=0, stop, step=1, dtype=None)
arange = np.arange(10, 20, 2)
# linspace(start, stop, num, endpoint=True, retstep=False, dtype=None)
linspaceWithoutRetstep = np.linspace(10, 20, 2)
linspaceWithRetstep = np.linspace(10, 20, 2, retstep=True)

print('empty\n', empty)
print('zeros\n', zeros)
print('ones\n', ones)
print('arange\n', arange)
print('linspaceWithoutRetstep\n', linspaceWithoutRetstep)
print('linspaceWithRetstep\n', linspaceWithRetstep)

# empty
#  [[ 9.90263869e+067  8.01304531e+262]
#  [ 2.60799828e-310 -4.30530469e+003]]
# zeros
#  [[0. 0.]
#  [0. 0.]]
# ones
#  [[1. 1.]
#  [1. 1.]]
# arange
#  [10 12 14 16 18]
# linspaceWithoutRetstep
#  [10. 20.]
# linspaceWithRetstep
#  (array([10., 20.]), 10.0)