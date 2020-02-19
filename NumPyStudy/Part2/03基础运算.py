import numpy as np

a = np.arange(1, 5)
b = np.arange(5, 9)

print('a--->\n', a)
print('b--->\n', b)

# 加减乘除
print('a + b = ', a + b)
print('a - b = ', a - b)
print('a * b = ', a * b)
print('a / b = ', a / b)
# 点乘
print('np.dot(a, b)\n', np.dot(a, b))
print('a.dot(b)\n', a.dot(b))
# 元素运算
x = np.random.randint(0, 10, (2, 5))
print('x--->\n', x)
print('np.sum(x)\n', np.sum(x))
print('np.sum(x, axis=0)\n', np.sum(x, axis=0))
print('np.min(x)\n', np.min(x))
print('np.min(x, axis=0)\n', np.min(x, axis=0))
print('np.max(x)\n', np.max(x))
print('np.max(x, axis=0)\n', np.max(x, axis=0))
# 最值索引
print('np.argmin(x)\n', np.argmin(x))
print('np.argmax(x)\n', np.argmax(x))
# 累加累减
print('np.cussum(x)\n', np.cumsum(x))
print('np.diff(x)\n', np.diff(x))
# 均值中位数
print('np.mean(x)\n', np.mean(x))
print('np.average(x)\n', np.average(x))
print('np.median(x)\n', np.median(x))
# 排序
print('np.sort(x)\n', np.sort(x))
# 转置
print('x.T\n', x.T)
print('np.transpose(x)\n', np.transpose(x))
# clip
print('np.clip(x, 3, 5)\n', np.clip(x, 3, 5))