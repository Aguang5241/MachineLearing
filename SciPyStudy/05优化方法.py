import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq


def main():
    # 定义拟合函数
    def func(p, x):
        w0, w1 = p
        f = w0 + w1 * x * x
        return f
    # 定义残差函数

    def err_func(p, x, y):
        ret = func(p, x) - y
        return ret

    p_int = np.random.randn(2)
    x = np.array([1, 2.5, 3.5, 4, 5, 7, 8.5])
    y = np.array([0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6])
    parameters = leastsq(err_func, p_int, args=(x, y))
    print(parameters[0])
    # [0.2092583  0.12013861]

    # 绘图验证
    plt.scatter(x, y)
    xx = np.linspace(0, 10, 100)
    yy = parameters[0][0] + parameters[0][1] * xx ** 2
    plt.plot(xx, yy)
    plt.show()


if __name__ == '__main__':
    main()
