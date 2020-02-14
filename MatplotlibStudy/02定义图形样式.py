import numpy as np
from matplotlib import pyplot as plt

######################1.绘制三角函数图形#########################
ax = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
ay1 = np.sin(ax)
ay2 = np.cos(ax)
# plt.plot(ax, ay1, color='#FF0000', linestyle='--', linewidth=2, alpha=0.8)
# plt.plot(ax, ay2, color='#000000', linestyle='-', linewidth=2)
# Fig.09
######################2.绘制散点图#########################
bx = np.random.rand(100)
by = np.random.rand(100)
colors = np.random.rand(100)
size = np.random.normal(50, 60, 10)
# plt.scatter(bx, by, s=size, c=colors)
# Fig.10
######################3.饼状图#########################
c = [1, 2, 3, 4, 5]
explode = (0, 0, 0, 0, 0.2)
labels = 'cat', 'dog', 'cattle', 'sheep', 'horse'
colors = 'r', 'g', 'r', 'g', 'y'
plt.pie(c, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
plt.show()