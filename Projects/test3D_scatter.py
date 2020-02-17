import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# x, y, z 均为 100 个随机数
x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
z = np.random.normal(0, 1, 100)

fig = plt.figure()

ax = Axes3D(fig)
ax.scatter(x, y, z)

plt.show()