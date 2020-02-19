import numpy as np
import matplotlib.pyplot as plt

def main():
    # parameters
    l = 1.0
    k = 1.0
    x0 = 1.0

    X = np.linspace(-10, 10, 1000)
    Y = np.zeros(1000)
    for i in range(1000):
        Y[i] = l / (1.0 + np.exp(-k * (X[i] - x0)))

    plt.title('Logistic Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(X, Y)
    plt.show()


if __name__ == '__main__':
    main()
