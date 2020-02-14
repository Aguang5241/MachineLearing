########################进阶应用############################
import numpy as np
###########51.创建一个 5x5 的二维数组，其中边界值为1，其余值为0##############
a = np.ones((5, 5))
a[1:4, 1:4] = 0
print(a)
# [[1. 1. 1. 1. 1.]
#  [1. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 1.]
#  [1. 1. 1. 1. 1.]]
###########52.使用数字 0 将一个全为 1 的 5x5 二维数组包围##############
# my solution
b1 = np.ones((7, 7))
b1[0] = b1[6, :] = b1[:, 0] = b1[:, 6] = 0
print(b1)
# the answer
b2 = np.ones((5, 5))
b2 = np.pad(b2, pad_width=1, mode='constant', constant_values=0)
print(b2)
# [[0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 1. 1. 1. 1. 0.]
#  [0. 1. 1. 1. 1. 1. 0.]
#  [0. 1. 1. 1. 1. 1. 0.]
#  [0. 1. 1. 1. 1. 1. 0.]
#  [0. 1. 1. 1. 1. 1. 0.]
#  [0. 0. 0. 0. 0. 0. 0.]]
###########关于np.diag()##############
c1 = np.arange(3)
c2 = np.arange(9).reshape(3, 3)
print(c1)
# [0 1 2]
print(c2)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]
print(np.diag(c1))
# [[0 0 0]
#  [0 1 0]
#  [0 0 2]]
print(np.diag(c2))
# [0 4 8]
###########53. 创建一个 5x5 的二维数组，并设置值 1, 2, 3, 4 落在其对角线下方##############
print(np.diag(np.arange(4) + 1, k=-1))
# [[0 0 0 0 0]
#  [1 0 0 0 0]
#  [0 2 0 0 0]
#  [0 0 3 0 0]
#  [0 0 0 4 0]]
###########54.创建一个 10x10 的二维数组，并使得 1 和 0 沿对角线间隔放置##############
d = np.zeros((10, 10), dtype=int)
d[1::2, ::2] = 1
d[::2, 1::2] = 1
print(d)
# [[0 1 0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0 1 0]]
###########55.创建一个 0-10 的一维数组，并将 (1, 9] 之间的数全部反转成负数##############
e = np.arange(11)
e[(e > 1) & (e <= 9)] *= -1
print(e)
# [ 0  1 -2 -3 -4 -5 -6 -7 -8 -9 10]
###########56.找出两个一维数组中相同的元素##############
f1 = np.random.randint(0, 10, 10)
f2 = np.random.randint(0, 10, 10)
print(f1)
# [2 1 5 3 1 6 7 1 3 0]
print(f2)
# [1 4 1 5 3 0 6 5 5 4]
print(np.intersect1d(f1, f2))
# [0 1 3 5 6]
###########57.使用 NumPy 打印昨天、今天、明天的日期##############
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today = np.datetime64('today', 'D')
tomorrow = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print("yesterday: ", yesterday)
# yesterday:  2020-02-11
print("today: ", today)
# today:  2020-02-12
print("tomorrow: ", tomorrow)
# tomorrow:  2020-02-13
###########58.使用五种不同的方法去提取一个随机数组的整数部分##############
g = np.random.uniform(0, 10, 10)
print("原始值: ", g)
# 原始值:  [4.09419929 1.94149353 0.13721603 2.47006423 8.857896   0.03618758
#  2.25144956 4.00889407 3.39157692 9.65151154]
print("方法 1: ", g - g % 1)
# 方法 1:  [4. 1. 0. 2. 8. 0. 2. 4. 3. 9.]
print("方法 2: ", np.floor(g))
# 方法 2:  [4. 1. 0. 2. 8. 0. 2. 4. 3. 9.]
print("方法 3: ", np.ceil(g)-1)
# 方法 3:  [4. 1. 0. 2. 8. 0. 2. 4. 3. 9.]
print("方法 4: ", g.astype(int))
# 方法 4:  [4 1 0 2 8 0 2 4 3 9]
print("方法 5: ", np.trunc(g))
# 方法 5:  [4. 1. 0. 2. 8. 0. 2. 4. 3. 9.]
###########59.创建一个 5x5 的矩阵，其中每行的数值范围从 1 到 5##############
# my solution
h1 = np.random.randint(1, 6, size=(5, 5), dtype=int)
print(h1)
# [[1 5 1 1 1]
#  [4 1 5 3 1]
#  [5 3 2 4 1]
#  [1 4 4 4 1]
#  [3 3 5 4 5]]
# the answer
h2 = np.zeros((5, 5), dtype=int)
h2 += np.arange(1, 6)
print(h2)
# [[1 2 3 4 5]
#  [1 2 3 4 5]
#  [1 2 3 4 5]
#  [1 2 3 4 5]
#  [1 2 3 4 5]]
###########60.创建一个长度为 5 的等间隔一维数组，其值域范围从 0 到 1，但是不包括 0 和 1##############
i = np.linspace(0, 1, 6, endpoint=False)[1:]
print(i)
# [0.16666667 0.33333333 0.5        0.66666667 0.83333333]
###########61.创建一个长度为10的随机一维数组，并将其按升序排序##############
# my solution
j1 = np.random.randint(0, 10, 10)
print(np.sort(j1))
# [0 0 1 1 2 2 5 5 7 7]
# the answer
j2 = np.random.randint(0, 10, 10)
j2.sort()
print(j2)
# [0 0 1 1 2 2 5 5 7 7]
###########62.创建一个 3x3 的二维数组，并将列按升序排序##############
# my solution
k1 = np.random.randint(10, size=(3, 3))
print(np.sort(k1, axis=0))
# the answer
k2 = np.random.randint(10, size=(3, 3))
k2.sort(axis=0)
print(k2)
# [[5 4 0]
#  [8 4 1]
#  [8 5 8]]
###########63.创建一个长度为 5 的一维数组，并将其中最大值替换成 0##############
l = np.random.randint(0, 10, 5)
print('Original Array:', l)
# Original Array: [4 4 6 5 8]
l[np.argmax(l)] = 0
print('Changed Array:', l)
# Changed Array: [4 4 6 5 0]
###########64.打印每个 NumPy 标量类型的最小值和最大值##############
for dtype in [np.int8, np.int32, np.int64]:
    print("The minimum value of {}: ".format(dtype), np.iinfo(dtype).min)
    print("The maximum value of {}: ".format(dtype), np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
    print("The minimum value of {}: ".format(dtype), np.finfo(dtype).min)
    print("The maximum value of {}: ".format(dtype), np.finfo(dtype).max)
# The minimum value of <class 'numpy.int8'>:  -128
# The maximum value of <class 'numpy.int8'>:  127
# The minimum value of <class 'numpy.int32'>:  -2147483648
# The maximum value of <class 'numpy.int32'>:  2147483647
# The minimum value of <class 'numpy.int64'>:  -9223372036854775808
# The maximum value of <class 'numpy.int64'>:  9223372036854775807
# The minimum value of <class 'numpy.float32'>:  -3.4028235e+38
# The maximum value of <class 'numpy.float32'>:  3.4028235e+38
# The minimum value of <class 'numpy.float64'>:  -1.7976931348623157e+308
# The maximum value of <class 'numpy.float64'>:  1.7976931348623157e+308
###########65.将 float32 转换为整型##############
m = np.arange(10, dtype=np.float32)
print('Original Array: ', m)
# Original Array:  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
m = m.astype(np.int32, copy=False)
print('Changed Array: ', m)
# Changed Array:  [0 1 2 3 4 5 6 7 8 9]
###########66.将随机二维数组按照第 3 列从上到下进行升序排列##############
n = np.random.randint(0, 10, size=(3, 3))
print('Original Array: \n', n)
# Original Array:
#  [[7 5 8]
#  [4 4 0]
#  [5 2 2]]
print('Changed Array: \n', n[n[:, 2].argsort()])
# Changed Array:
#  [[4 4 0]
#  [5 2 2]
#  [7 5 8]]
###########67.从随机一维数组中找出距离给定数值（0.5）最近的数##############
o = np.random.uniform(0, 1, 10)
print('Random Array: \n', o)
# Random Array:
#  [0.87604093 0.60815843 0.74404004 0.7476415  0.39862474 0.14744497
#  0.48903291 0.28790076 0.93716727 0.9511766 ]
target = 0.5
result = o.flat[np.abs(o - target).argmin()]
print('Result: ', result)
# Result:  0.48903290555618106
###########68.将二维数组的前两行进行顺序交换##############
p = np.random.randint(0, 10, size=(3, 3))
print('Original Array: \n', p)
# Original Array:
#  [[7 7 9]
#  [0 5 0]
#  [6 0 5]]
p[[0, 1]] = p[[1, 0]]
print('Changed Array: \n', p)
# Changed Array:
#  [[0 5 0]
#  [7 7 9]
#  [6 0 5]]
###########69.找出随机一维数组中出现频率最高的值##############
q = np.random.randint(0, 3, 10)
print('Original Array: ', q)
print('Result: ', np.bincount(q).argmax())
# Original Array:  [2 0 2 1 1 0 1 1 1 2]
# Result:  1
###########70.找出给定一维数组中非 0 元素的位置索引##############
# my solution
r = np.random.randint(0, 5, 10)
print('Original Array: ', r)
# Original Array:  [4 4 2 2 4 2 0 0 2 2]
indexes = []
jj = 0
for ii in r:
	jj += 1
	if ii != 0:
		indexes.append(jj - 1)
print('Result: ', indexes)
# Result:  [0, 1, 2, 3, 4, 5, 8, 9]
# the answer
print('Result: ', np.nonzero(r))
# Result:  (array([0, 4, 5, 6, 7, 9], dtype=int64),)
###########71.对于给定的 5x5 二维数组，在其内部随机放置 p 个值为 1 的数##############
s = np.zeros((5, 5))
np.put(s, np.random.choice(range(5*5), 3, replace=False), 1)
print(s)
# [[0. 0. 0. 0. 1.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]]
###########72.对于随机的 3x3 二维数组，减去数组每一行的平均值##############
t = np.random.randint(0, 10, (5, 5))
meanValue = np.mean(t, axis=0)
print(t)
# [[5 1 0 1 8]
#  [6 0 6 0 7]
#  [0 4 7 2 1]
#  [7 2 8 1 6]
#  [8 2 5 0 4]]
print(meanValue)
# [5.2 1.8 5.2 0.8 5.2]
print(t - meanValue)
# [[-0.2 -0.8 -5.2  0.2  2.8]
#  [ 0.8 -1.8  0.8 -0.8  1.8]
#  [-5.2  2.2  1.8  1.2 -4.2]
#  [ 1.8  0.2  2.8  0.2  0.8]
#  [ 2.8  0.2 -0.2 -0.8 -1.2]]
###########73.获得二维数组点积结果的对角线数组##############
# 普通方法
q1 = np.random.randint(0, 10, (3, 3))
q2 = np.random.randint(0, 10, (3, 3))
dotValue = np.dot(q1, q2)
print(dotValue)
# [[ 93 103  81]
#  [ 48  61  27]
#  [ 33  21  12]]
print(np.diag(dotValue))
# 较快的方法
print(np.sum(q1 * q2.T, axis=1))
# 更快的方法
np.einsum("ij, ji->i", q1, q2)
# [93 61 12]
###########74.找到随机一维数组中前 p 个最大值##############
r = np.random.randint(0, 30, 10)
print(r)
# [ 8 14 27 22 29 10 24 29  5 23]
print(r[np.argsort(r)[-3:]])
# [27 29 29]
###########75.计算随机一维数组中每个元素的 4 次方数值##############
s = np.random.randint(1, 10, 5)
print(s)
# [1 9 9 6 7]
print(np.power(s, 4))
# [1 6561 6561 1296 2401]
###########76.对于二维随机数组中各元素，保留其 2 位小数##############
t = np.random.uniform(0, 10, (3, 3))
print(np.round_(t, 2))
# [[3.49 2.   5.36]
#  [5.56 9.65 4.18]
#  [0.21 1.82 8.7 ]]
###########77.使用科学记数法输出 NumPy 数组##############
u = np.random.random_sample([3, 3])
print(u / 1e3)
# [[6.34270472e-04 9.03450397e-04 4.84688574e-05]
#  [5.05325392e-04 4.43503753e-04 4.76787496e-04]
#  [3.72711398e-04 1.43364917e-04 8.62723066e-04]]
###########78.使用 NumPy 找出百分位数（25%，50%，75%）##############
v = np.arange(15)
print(np.percentile(v, q=[25, 50, 75]))
# [ 3.5  7.  10.5]
###########79.找出数组中缺失值的总数及所在位置##############
w = np.random.rand(10, 10)
w[np.random.randint(0, 10, 5), np.random.randint(0, 10, 5)] = np.nan
print(np.isnan(w).sum())
# 5
print(np.where(np.isnan(w)))
# (array([0, 1, 3, 5, 5], dtype=int64), array([4, 7, 5, 5, 6], dtype=int64))
###########80.从随机数组中删除包含缺失值的行##############
w[np.sum(np.isnan(w), axis=1) == 0]
print(w)