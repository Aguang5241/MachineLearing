import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 10, 20)
y = x * x + 2

fig = plt.figure()
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # 左、下、宽、高占比
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])
axes1.plot(x, y, 'r')
axes2.plot(y, x, 'g')
# Fig.13
plt.show()