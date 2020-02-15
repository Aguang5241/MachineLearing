# x ^ x 函数

import math
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    x = np.linspace(0.0001, 1.3, 101)
    y = x ** x
    plt.plot(x, y, 'g-', label = 'x^x', linewidth = 2)
    plt.grid()
    plt.legend(loc = 'upper left')
    plt.show()