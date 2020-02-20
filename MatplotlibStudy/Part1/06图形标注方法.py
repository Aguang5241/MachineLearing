import numpy as np
from matplotlib import pyplot as plt

###################文字标注######################
# fig, axes = plt.subplots()

# x_bar = [10, 20, 30, 40, 50]  # 柱形图横坐标
# y_bar = [0.5, 0.6, 0.3, 0.4, 0.8]  # 柱形图纵坐标
# bars = axes.bar(x_bar, y_bar, color='blue', label=x_bar, width=2)  # 绘制柱形图
# for i, rect in enumerate(bars):
#     x_text = rect.get_x()  # 获取柱形图横坐标
#     y_text = rect.get_height() + 0.01  # 获取柱子的高度并增加 0.01
#     plt.text(x_text, y_text, '%.1f' % y_bar[i])  # 标注文字
# plt.show()
# Fig.23
###################符号标注######################
fig, axes = plt.subplots()
x_bar = [10, 20, 30, 40, 50]  # 柱形图横坐标
y_bar = [0.5, 0.6, 0.3, 0.4, 0.8]  # 柱形图纵坐标
bars = axes.bar(x_bar, y_bar, color='blue', label=x_bar, width=2)  # 绘制柱形图
for i, rect in enumerate(bars):
    x_text = rect.get_x()  # 获取柱形图横坐标
    y_text = rect.get_height() + 0.01  # 获取柱子的高度并增加 0.01
    plt.text(x_text, y_text, '%.1f' % y_bar[i])  # 标注文字

    # 增加箭头标注
    plt.annotate('Min', xy=(32, 0.3), xytext=(36, 0.3),
                 arrowprops=dict(facecolor='black', width=1, headwidth=7))
plt.show()