import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y1 = 2 * x + 1
y2 = x ** 2

plt.figure()
plt.plot(x, y1)

plt.figure()
plt.plot(x, y2, color='red', linewidth=1.5, linestyle='--')

plt.show()