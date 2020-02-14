import numpy as np
from matplotlib import pyplot as plt

# ###################1.添加图标题、图例#####################
x = np.linspace(0, 2, 100)
# fig, axes = plt.subplots()
# axes.set_xlabel('x')
# axes.set_ylabel('y')
# axes.set_title('Figure_14')
# axes.plot(x, x ** 2)
# axes.plot(x, x ** 3)
# axes.legend(['y = x ** 2', 'y = x ** 3'], loc=0)
# plt.show()
# ###################2.线型、颜色、透明度#####################
# 颜色、透明度
# fig, axes = plt.subplots()
# axes.plot(x, x + 1, color='red', alpha=0.3)
# axes.plot(x, x + 2, color='#1155dd', alpha=0.5)
# axes.plot(x, x + 3, color='#15cc15')
# plt.show()
# Fig.15
# 线宽
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(x, x+1, color="blue", linewidth=0.25)
# ax.plot(x, x+2, color="blue", linewidth=0.50)
# ax.plot(x, x+3, color="blue", linewidth=1.00)
# ax.plot(x, x+4, color="blue", linewidth=2.00)
# plt.show()
# Fig.16
# 虚线类型
# ax.plot(x, x+5, color="red", lw=2, linestyle='-')
# ax.plot(x, x+6, color="red", lw=2, ls='-.')
# ax.plot(x, x+7, color="red", lw=2, ls=':')
# plt.show()
# Fig.17
# 虚线交错宽度
# line, = ax.plot(x, x+8, color="black", lw=1.50)
# line.set_dashes([5, 10, 15, 10])
# plt.show()
# Fig.18
# 符号
# ax.plot(x, x + 9, color="green", lw=2, ls='--', marker='+')
# ax.plot(x, x+10, color="green", lw=2, ls='--', marker='o')
# ax.plot(x, x+11, color="green", lw=2, ls='--', marker='s')
# ax.plot(x, x+12, color="green", lw=2, ls='--', marker='1')
# plt.show()
# Fig.19
# 符号大小和颜色
# ax.plot(x, x+13, color="purple", lw=1, ls='-', marker='o', markersize=2)
# ax.plot(x, x+14, color="purple", lw=1, ls='-', marker='o', markersize=4)
# ax.plot(x, x+15, color="purple", lw=1, ls='-',
#         marker='o', markersize=8, markerfacecolor="red")
# ax.plot(x, x+16, color="purple", lw=1, ls='-', marker='s', markersize=8,
#         markerfacecolor="yellow", markeredgewidth=2, markeredgecolor="blue")
# plt.show()
# Fig.20
# ###################3.画布网格、坐标轴范围#####################
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# # 显示网格
# axes[0].plot(x, x**2, x, x**3, lw=2)
# axes[0].grid(True)
# # 设置坐标轴范围
# axes[1].plot(x, x**2, x, x**3)
# axes[1].set_ylim([0, 60])
# axes[1].set_xlim([2, 5])
# plt.show()
# Fig.21
# 综合图
n = np.array([0, 1, 2, 3, 4, 5])
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
axes[0].scatter(x, x + 0.25*np.random.randn(len(x)))
axes[0].set_title("scatter")
axes[1].step(n, n**2, lw=2)
axes[1].set_title("step")
axes[2].bar(n, n**2, align="center", width=0.5, alpha=0.5)
axes[2].set_title("bar")
axes[3].fill_between(x, x**2, x**3, color="green", alpha=0.5)
axes[3].set_title("fill_between")
plt.show()
# Fig.22