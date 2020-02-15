from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def main():
    t = np.linspace(-1, 1, 100)

    def f(t):
        return np.sin(np.pi*t) + 0.1*np.cos(7*np.pi*t+0.3) + 0.2 * np.cos(24*np.pi*t) + 0.3*np.cos(12*np.pi*t+0.5)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    axes[0].plot(t, signal.gausspulse(t, fc=5, bw=0.5))
    axes[0].set_title("gausspulse")
    t *= 5*np.pi
    axes[1].plot(t, signal.sawtooth(t))
    axes[1].set_title("chirp")
    axes[2].plot(t, signal.square(t))
    axes[2].set_title("gausspulse")

    t = np.linspace(0, 4, 400)
    plt.plot(t, f(t))
    # 中值滤波函数
    plt.plot(t, signal.medfilt(f(t), kernel_size=55), linewidth=2, label="medfilt")
    # 维纳滤波函数
    plt.plot(t, signal.wiener(f(t), mysize=55), linewidth=2, label="wiener")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
