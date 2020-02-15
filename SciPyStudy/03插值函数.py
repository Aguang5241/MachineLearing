import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])

# 两点之间的点的 x 坐标
xx = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
# 使用原样本点建立插值函数
f = interpolate.interp1d(x, y)
yy = f(xx)

plt.scatter(x, y)
plt.scatter(xx, yy, marker='*')

plt.show()
