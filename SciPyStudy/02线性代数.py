from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt

#################1.矩阵的逆####################
a = np.matrix([[1, 2], [3, 4]])
print(a)
# [[1 2]
#  [3 4]]
print(linalg.inv(a))
# [[-2.   1. ]
#  [ 1.5 -0.5]]
#################2.奇异值分解####################
b = np.random.randint(0, 10, size=(3, 2))
print(b)
# [[6 9]
#  [0 3]
#  [7 2]]
Ub, sb, Vhb = linalg.svd(b)
print('the Ub:\n', Ub)
# the Ub:
#  [[-0.84987557 -0.38300844 -0.36196138]
#  [-0.17444003 -0.44367038  0.87904907]
#  [-0.49727476  0.81022289  0.31025261]]
print('the sb:\n', sb)
# the sb:
#  [12.55582472  4.62074297]
print('the Vhb:\n', Vhb)
# the Vhb:
#  [[-0.68336226 -0.73007946]
#  [ 0.73007946 -0.68336226]]
#################3.最小二乘法求解函数####################
x = np.array([1, 2.5, 3.5, 4, 5, 7, 8.5])
y = np.array([0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6])
M = x[:, np.newaxis]**[0, 2]
print(M)
# [[ 1.    1.  ]
#  [ 1.    6.25]
#  [ 1.   12.25]
#  [ 1.   16.  ]
#  [ 1.   25.  ]
#  [ 1.   49.  ]
#  [ 1.   72.25]]
p = linalg.lstsq(M, y)[0]
print(p)
# [0.20925829 0.12013861]
# 绘图验证
plt.scatter(x, y)
xx = np.linspace(0, 10, 100)
yy = p[0] + p[1]*xx**2
plt.plot(xx, yy)
plt.show()