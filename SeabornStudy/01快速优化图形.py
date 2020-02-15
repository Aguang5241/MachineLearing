import matplotlib.pyplot as plt
import seaborn as sns

x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
y_bar = [3, 4, 6, 8, 9, 10, 9, 11, 7, 8]
y_line = [2, 3, 5, 7, 8, 9, 8, 10, 6, 7]

# 只需要在绘图前放置 seaborn 提供的样式声明代码
sns.set()
plt.bar(x, y_bar)
plt.plot(x, y_line, '-o', color='y')
plt.show()
# fig.02