import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def draw(x, y):
    sns.set(style='ticks', font='Arial', font_scale=1.5)
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(x, y)
    plt.savefig(r'D:\Study\Coding\MachineLearing\DeepLearningWithPyTorch\res\2.2.4.rawdata.png')
    plt.show()

def get_data():
    train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.336, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    dtype = torch.FloatTensor
    X = torch.from_numpy(train_X).type(dtype=dtype).view(17, 1)
    y = torch.from_numpy(train_Y).type(dtype=dtype)
    return X, y

def get_weights():
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    return w, b


def main():
    x, y = get_data()
    w, b = get_weights()
    draw(x.numpy(), y.numpy())
    
    # for i in range(500):
    #     y_pred = simple_network(x)
    #     loss = loss_fn(y, y_pred)
    #     if i % 50 == 0:
    #         print(loss)
    #     optimize(learning_rate)

if __name__ == '__main__':
    main()