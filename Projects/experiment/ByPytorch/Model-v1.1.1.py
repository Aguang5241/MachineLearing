###########################################################
#                net shape: 5-10-1                        #
#                layer function: Linear                   #
#                standard: True                           #
#                activation function: ReLU                #
#                loss function: MSELoss                   #
#                optimizer: Adam                          #
###########################################################

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

# 学习率
learning_rate = 1e-3
# 设置损失阈值
loss_threashold_value = 1e-3
# 设置最大循环数
loop_max = 100000
# 设置保存路径（带标记）
index = np.random.randn(1)
path = 'Projects/Experiment/res/model-v1.1.1/Part1/%.3f/' % index
# 设置训练及测试数据路径
training_data_file_path = 'Projects/Experiment/res/TrainingData.csv'
testing_data_file_path = 'Projects/Experiment/res/TestingData.csv'


# 定义神经网络
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


# 定义获取测试数据函数
def getTestingData(file_path):
    data = pd.read_csv(file_path)
    x = torch.from_numpy(data.loc[:, 'PH_Al':'PH_Mg2Si'].values).float()
    EL_Si = torch.unsqueeze(
        (torch.from_numpy(data['EL_Si'].values)), dim=1).float()
    EL_Mg = torch.unsqueeze(
        (torch.from_numpy(data['EL_Mg'].values)), dim=1).float()
    return x, EL_Si, EL_Mg


# 定义获取训练数据函数
def getTrainingData(file_path):
    data = pd.read_csv(file_path)
    x = torch.from_numpy(data.loc[:, 'PH_Al':'PH_Mg2Si'].values).float()
    # print(x.shape) # (6 ,5)
    y_UTS = torch.unsqueeze(
        (torch.from_numpy(data['UTS'].values)), dim=1).float()
    y_YS = torch.unsqueeze(
        (torch.from_numpy(data['YS'].values)), dim=1).float()
    y_EL = torch.unsqueeze(
        (torch.from_numpy(data['EL'].values)), dim=1).float()
    # print(y_*.shape) # (6, 1)
    EL_Si = torch.unsqueeze(
        (torch.from_numpy(data['EL_Si'].values)), dim=1).float()
    EL_Mg = torch.unsqueeze(
        (torch.from_numpy(data['EL_Mg'].values)), dim=1).float()
    return x, y_UTS, y_YS, y_EL, EL_Si, EL_Mg


# 定义训练函数
def train(x, y):
    # 实例化神经网络
    net = Net(n_feature=5, n_hidden=10, n_output=1)
    # Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # 损失函数（均方差）
    loss_func = torch.nn.MSELoss()
    # 训练神经网络
    # 初始化
    loop = 0
    training_break = False
    start_time = time.time()
    predict_y = net(x)
    loss_y = loss_func(predict_y, y)
    # 循环训练
    while loss_y > loss_threashold_value:
        loop += 1
        predict_y = net(x)
        loss_y = loss_func(predict_y, y)
        optimizer.zero_grad()
        loss_y.backward()
        optimizer.step()
        if (loop <= loop_max):
            if (loop % 1000 == 0):
                print('Loop: %dK ---' % (loop / 1000),
                      'loss: %.6f' % loss_y.item())
        else:
            user_choice = input('Continue or not(Y/N)')
            if (user_choice.lower() != 'y'):
                training_break = True
                print('Training break!!!')
                break
            else:
                loop = 0

    if not training_break:
        os.makedirs(path)
        torch.save(
            net, path + 'model-v1.1.1.pkl')

    end_time = time.time()
    print('Total time: %.2fs' % (end_time - start_time))
    return training_break


# 定义测试函数
def test(model_path, x):
    net = torch.load(model_path)
    predict_y = net(x)
    return predict_y


# 绘制散点图
def draw_scatter(x_training, y_training, z_training, x_testing, y_testing, z_testing):
    sns.set(font="Times New Roman", font_scale=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('Mg')
    ax.set_ylabel('Si')
    ax.set_zlabel('Performance')
    ax.scatter(x_training, y_training, z_training, color='red', s=50)
    ax.scatter(x_testing, y_testing, z_testing)
    plt.savefig(path + 'model-v1.1.1.png')
    plt.show()


def main():
    # 获取训练及测试数据
    x, y_UTS, y_YS, y_EL, EL_Si, EL_Mg = getTrainingData(
        training_data_file_path)
    x_testing, EL_Si_test, EL_Mg_test = getTestingData(testing_data_file_path)

    # 执行正则化，并记住训练集数据的正则化规则,运用于测试集数据
    x_scaler = StandardScaler().fit(x)
    # x_scaler = StandardScaler().fit(x_testing)
    x_standarded = torch.from_numpy(x_scaler.transform(x)).float()
    x_standarded_test = torch.from_numpy(x_scaler.transform(x_testing)).float()

    # 执行模型训练
    training_break = train(x_standarded, y_UTS)

    # 调用训练好的模型进行预测
    if not training_break:
        model_path = path + 'model-v1.1.1.pkl'
        # 此处不需要跟踪梯度
        with torch.no_grad():
            y_testing = test(model_path, x_standarded_test)
            if np.isnan(y_testing.numpy().any()):
                print('Run out of range!')
            else:
                print('Just fine, index ---> %.3f' % index)

        # 数据可视化(散点图)
        draw_scatter(EL_Si.numpy(), EL_Mg.numpy(), y_UTS.numpy(),
                     EL_Si_test.numpy(), EL_Mg_test.numpy(), y_testing.numpy())


if __name__ == '__main__':
    main()
