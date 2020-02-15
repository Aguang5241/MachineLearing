import numpy as np
from matplotlib import pyplot as plt 
import matplotlib

x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]

x = np.arange(-200, -100, 1) # bias
y = np.arange(-5, 5, 0.1) # weight
Z = np.zeros((len(x), len(y)))
X, Y = np.meshgrid(x, y)
for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[i][j] = 0
        for n in range(len(x_data)):
            Z[i][j] = Z[i][j] + (y_data[n] - b * w * x_data[n] ** 2)
        Z[i][j] = Z[i][j] / len(x_data)

# ydata = b + w * xdata
b = -120 # initial b
w = -4 # initail w
lr = 1 # learning rate
iteration = 100000

# store initial values for plotting
b_histroy = [b]
w_history = [w]

lr_b = 0.0
lr_w = 0.0

# iteration
for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * 1.0
        w_grad = w_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]

    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2

    # update parameters
    b = b - lr / np.sqrt(lr_b) * b_grad
    w = w - lr / np.sqrt(lr_w) * w_grad

    # store parameters for plotting
    b_histroy.append(b)
    w_history.append(w)

# plotting
plt.contourf(x, y, Z, 50, alpha=0.5, camp=plt.get_cmap('jet'))
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
plt.plot(b_histroy, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()