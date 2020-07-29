import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    # 绘制原始散点
    def draw(x, y):
        sns.set(style='ticks', font='Arial', font_scale=1.5)
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(x, y)
        plt.savefig(
            r'D:\Study\Coding\MachineLearing\DeepLearningWithPyTorch\res\2.2.4.rawdata.png')
        # plt.show()

    # 绘制拟合结果
    def draw_(x, y, w, b):
        x_ = np.linspace(0, 11, 100)
        sns.set(style='ticks', font='Arial', font_scale=1.5)
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(x, y)
        plt.plot(x_, w * x_ + b, color='r')
        plt.savefig(
            r'D:\Study\Coding\MachineLearing\DeepLearningWithPyTorch\res\2.2.4.fitdata.png')
        # plt.show()

    # 获取散点数据（张量）
    def get_data():
        train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182,
                              7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
        train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.336,
                              2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
        dtype = torch.FloatTensor
        X = torch.from_numpy(train_X).type(dtype=dtype).view(17, 1)
        y = torch.from_numpy(train_Y).type(dtype=dtype)
        return X, y

    # 获取权重和偏置（变量）
    def get_weights():
        w = torch.randn(1, requires_grad=True)
        b = torch.randn(1, requires_grad=True)
        return w, b

    # 定义简单的训练模型（y=wx+b）
    def simple_network(x):
        y_pred = torch.matmul(x, w) + b
        return y_pred

    # 计算损失值
    def loss_fn(y, y_pred):
        loss = (y - y_pred).pow(2).sum()
        for param in [w, b]:
            if not param.grad is None:
                param.grad.data.zero_()
            loss.backward()
            return loss.data

    # 优化函数
    def optimize(learning_rate):
        w.data -= learning_rate * w.grad.data
        b.data -= learning_rate * b.grad.data

    x, y = get_data()
    w, b = get_weights()
    learning_rate = 1e-3

    draw(x.numpy(), y.numpy())

    for i in range(1000):
        y_pred = simple_network(x)
        loss = loss_fn(y, y_pred)
        if i % 50 == 0:
            print(loss)
        optimize(learning_rate)

    draw_(x.numpy(), y.numpy(), w.detach().numpy(), b.detach().numpy())


if __name__ == '__main__':
    main()
