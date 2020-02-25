import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    # 定义神经网络
    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)
            self.predict = torch.nn.Linear(n_hidden, n_output)

        def forward(self, x):
            # 激励函数
            x = F.relu(self.hidden(x))
            x = self.predict(x)
            return x

    # 定义获取原始数据函数
    def getData(file_path):
        data = pd.read_csv(file_path)
        x = torch.from_numpy(data.loc[:, 'PH_Al':'PH_AlSc2Si2'].values).float()
        # print(x.shape) # (6 ,4)
        y_UTS = torch.unsqueeze(
            (torch.from_numpy(data['UTS'].values)), dim=1).float()
        y_YS = torch.unsqueeze(
            (torch.from_numpy(data['YS'].values)), dim=1).float()
        y_EL = torch.unsqueeze(
            (torch.from_numpy(data['EL'].values)), dim=1).float()
        # print(y_*.shape) # (6, 1)
        return x, y_UTS, y_YS, y_EL

    # 获取原始数据
    file_path = 'Projects/Experiment/res/filteredData_withoutElements.csv'
    x, y_UTS, y_YS, y_EL = getData(file_path)
    print(x)
    # 四个特征值（四个相的相分数）+一个输出值（力学性能UTS/YS/EL）
    # 实例化神经网络
    net = Net(n_feature=4, n_hidden=10, n_output=1)
    # 学习率
    learning_rate = 0.1
    # print(net)
    # Net(
    # (hidden): Linear(in_features=4, out_features=10, bias=True)
    # (predict): Linear(in_features=10, out_features=1, bias=True)
    # )
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    # 损失函数（均方差）
    loss_func = torch.nn.MSELoss()
    # 训练网络
    # for i in range(500):
    #     predict_y = net(x)
    #     loss_UTS = loss_func(predict_y, y_UTS)
    #     # loss_YS = loss_func(predict_y, y_YS)
    #     # loss_EL = loss_func(predict_y, y_EL)
    #     optimizer.zero_grad()
    #     loss_UTS.backward()
    #     # loss_YS.backward()
    #     # loss_EL.backward()
    #     optimizer.step()
    #     if (i % 5 == 0):
    #         print(i, loss_UTS)
            

if __name__ == '__main__':
    main()
