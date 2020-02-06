import numpy as np

a = np.array([1.1, 2.2, 3.3], dtype=np.float64) # 指定一维数组及其类型

##############查看a的dtype类型######################
print(a)
# [1.1 2.2 3.3]
print(a.dtype)
# float64
##############使用.astype(obj)转换数值类型###########
print(a.astype(int).dtype)
# int32
######################创建数组######################
###1.np.array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0)###
b = np.array([[1, 2, 3], [4, 5, 6]])
c = np.array([(1, 2, 3), (4, 5, 6)])
print(b)
print(c)
# [[1 2 3]
#  [4 5 6]]
###2.np.arange(start, stop, step, dtype=None)###
d = np.arange(0, 7, 0.5, dtype='float32')
print(d)
# [0.  0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5.  5.5 6.  6.5]
###3.np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)###

