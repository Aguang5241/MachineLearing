###########################################################
#                net shape: 4-10-5-3                      #
#                layer function: Linear                   #
#                standard: True                           #
#                activation function: ReLU                #
#                loss function: MSELoss                   #
#                optimizer: Adam                          #
#                visualize the loss                       #
#                0.5UTS + 0.5YS + EL                      #
#                output the coefficients                  #
#                output the results                       #
#                visualize the results                    #
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 设置学习率
learning_rate = 1e-4
# 设置损失阈值
loss_threashold_value = 1e-2
# 设置误差矩阵
e = torch.tensor([3, 3, 0.1]).float()
error = e.repeat(6, 1)
# 设置单次最大循环数
loop_max = 100000
# 设置保存路径（带标记）
index = np.random.randn(1)
path = 'Projects/Experiment_/res/model-v1.5/%.3f/' % index
# 设置训练及测试数据路径
training_data_file_path = 'Projects/Experiment_/res/Data/TrainingData.csv'
testing_data_file_path = 'Projects/Experiment_/res/Data/TestingDataFiltered.csv'


# 定义神经网络
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x


# 定义获取测试数据函数
def get_testing_data(file_path):
    data = pd.read_csv(file_path)
    x = torch.from_numpy(data.loc[:, 'PH_Al':'PH_AlSc2Si2'].values).float()
    EL_Si = torch.unsqueeze(
        (torch.from_numpy(data['EL_Si'].values)), dim=1).float()
    EL_Mg = torch.unsqueeze(
        (torch.from_numpy(data['EL_Mg'].values)), dim=1).float()
    return x, EL_Si, EL_Mg


# 定义获取训练数据函数
def get_training_data(file_path):
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
    EL_Si = torch.unsqueeze(
        (torch.from_numpy(data['EL_Si'].values)), dim=1).float()
    EL_Mg = torch.unsqueeze(
        (torch.from_numpy(data['EL_Mg'].values)), dim=1).float()
    return x, y_UTS, y_YS, y_EL, EL_Si, EL_Mg


# 定义训练函数
def train(x, y):
    # 实例化神经网络
    net = Net(n_feature=4, n_hidden1=10, n_hidden2=5, n_output=3)
    # Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # 损失函数
    loss_func = torch.nn.MSELoss()
    # 数据初始化
    loop = 0
    training_break = False
    start_time = time.time()
    predict_y = net(x)
    loss_y = torch.abs(predict_y - y)
    loss = loss_func(loss_y, error)
    # 图像配置初始化
    sns.set(font="Times New Roman")
    fig = plt.figure()
    ax = plt.gca()
    plt.xlabel('Loops/K')
    plt.ylabel('Loss value')
    plt.ylim(-0.1, 3)
    # 循环训练
    while loss > loss_threashold_value:
        loop += 1
        predict_y = net(x)
        loss_y = torch.abs(predict_y - y)
        loss = loss_func(loss_y, error)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (loop <= loop_max):
            if (loop % 1000 == 0):
                print('Loop: %dK ---' % (loop / 1000),
                      'loss: %.6f' % loss.item())
                # 可视化误差变化
                if loss.item() <= 3:
                    ax.scatter(loop / 1000, loss.item(), color='red', s=5)
                    plt.pause(0.1)
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
        torch.save(net, path + 'model-v1.4.4.pkl')

    end_time = time.time()
    w_1 = net.hidden1.weight
    w_2 = net.hidden2.weight
    w_3 = net.predict.weight
    b_1 = net.hidden1.bias
    b_2 = net.hidden2.bias
    b_3 = net.predict.bias
    w_total = (w_3.mm(w_2)).mm(w_1)
    b_total = w_3.mm(w_2.mm(b_1.view(10, 1))) + \
        w_3.mm(b_2.view(5, 1)) + b_3.view(3, 1)
    print('===================Training complete====================')
    print('Total time: %.2fs' % (end_time - start_time))
    print('layer1 weight ---> ', w_1)
    print('layer1 bias ---> ', b_1)
    print('layer2 weight ---> ', w_2)
    print('layer2 bias ---> ', b_2)
    print('layer3 weight ---> ', w_3)
    print('layer3 bias ---> ', b_3)
    print('Total weight ---> \n', w_total)
    print('Total bias ---> \n', b_total)
    np.savetxt(path + 'w_1.csv',
               w_1.detach().numpy(), fmt='%.3f', delimiter=',')
    np.savetxt(path + 'w_2.csv',
               w_2.detach().numpy(), fmt='%.3f', delimiter=',')
    np.savetxt(path + 'w_3.csv',
               w_3.detach().numpy(), fmt='%.3f', delimiter=',')
    np.savetxt(path + 'b_1.csv',
               b_1.detach().numpy(), fmt='%.3f', delimiter=',')
    np.savetxt(path + 'b_2.csv',
               b_2.detach().numpy(), fmt='%.3f', delimiter=',')
    np.savetxt(path + 'b_3.csv',
               b_3.detach().numpy(), fmt='%.3f', delimiter=',')
    np.savetxt(path + 'total_weight.csv',
               w_total.detach().numpy(), fmt='%.3f', delimiter=',')
    np.savetxt(path + 'total_bias.csv', b_total.detach().numpy(),
               fmt='%.3f', delimiter=',')
    plt.savefig(path + 'learning_curve.png')
    plt.show()
    return training_break


# 定义测试函数
def test(model_path, x):
    net = torch.load(model_path)
    predict_y = net(x)
    pd.DataFrame(predict_y.numpy()).to_csv(
        path + 'testing_results.csv', index=False, header=['UTS', 'YS', 'EL'])
    return predict_y


# 综合处理全部数据
def data_process(path, x, y):
    data = pd.read_csv(path + 'testing_results.csv')
    mms = MinMaxScaler()
    data_processed = mms.fit_transform(data.values)
    # data_calculated = 0.5 * data_processed[:, 0] + 0.5 * data_processed[:, 1] + data_processed[:, 2]
    data_calculated = 0.5 * data_processed[:, 0] + 0.5 * data_processed[:, 1]
    # 获取最值索引
    max_index = data_calculated.tolist().index(max(data_calculated))
    print('========================Results=========================')
    results = 'Si: ' + str(x[max_index][0]) + \
        '\nMg: ' + str(y[max_index][0]) + \
        '\nUTS: ' + str(data.values[max_index, 0]) + \
        '\nYS: ' + str(data.values[max_index, 1]) + \
        '\nEL: ' + str(data.values[max_index, 2])
    print(results)
    f = open(path + 'optimal_general_performance.csv', 'w')
    f.write(results)
    f.close()
    # 可视化
    sns.set(font="Times New Roman")
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('Si')
    ax.set_ylabel('Mg')
    ax.set_zlabel('General Performance')
    ax.scatter(x, y, data_calculated)
    ax.scatter(x[max_index], y[max_index],
               data_calculated[max_index], color='red', s=50)
    plt.savefig(path + 'general_Performance.png')
    plt.show()


# 绘制散点图
def draw_scatter(x_training, y_training, z_training, x_testing, y_testing, z_testing, item):
    if item == 'UTS / MPa':
        fig_name = 'UTS'
    elif item == 'YS / MPa':
        fig_name = 'YS'
    else:
        fig_name = 'EL'
    sns.set(font="Times New Roman", font_scale=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('Si')
    ax.set_ylabel('Mg')
    ax.set_zlabel(item)
    ax.scatter(x_training, y_training, z_training, color='red', s=50)
    ax.scatter(x_testing, y_testing, z_testing)
    plt.savefig(path + 'elements_%s.png' % fig_name)
    plt.show()


# 绘制相分数-性能关系图
def draw_relation(x_training, y_training, x_testing, y_testing):
    sns.set(font="Times New Roman", font_scale=1)
    # UTS
    fig1, ax1 = plt.subplots(2, 2, figsize=(16, 12))
    # Al_1/UTS
    ax1[0][0].set_xlabel('Al_1 / wt.%')
    ax1[0][0].set_ylabel('UTS / MPa')
    ax1[0][0].set_title('Al_1/UTS', fontstyle='oblique')
    ax1[0][0].scatter(x_testing[:, 0], y_testing[:, 0])
    ax1[0][0].scatter(x_training[:, 0], y_training[:, 0], color='red')
    # Al_2/UTS
    ax1[0][1].set_xlabel('Al_2 / wt.%')
    ax1[0][1].set_ylabel('UTS / MPa')
    ax1[0][1].set_title('Al_2/UTS', fontstyle='oblique')
    ax1[0][1].scatter(x_testing[:, 1], y_testing[:, 0])
    ax1[0][1].scatter(x_training[:, 1], y_training[:, 0], color='red')
    # Si/UTS
    ax1[1][0].set_xlabel('Si / wt.%')
    ax1[1][0].set_ylabel('UTS / MPa')
    ax1[1][0].set_title('Si/UTS', fontstyle='oblique')
    ax1[1][0].scatter(x_testing[:, 2], y_testing[:, 0])
    ax1[1][0].scatter(x_training[:, 2], y_training[:, 0], color='red')
    # AlSc2Si2/UTS
    ax1[1][1].set_xlabel('AlSc2Si2 / wt.%')
    ax1[1][1].set_ylabel('UTS / MPa')
    ax1[1][1].set_title('AlSc2Si2/UTS', fontstyle='oblique')
    ax1[1][1].scatter(x_testing[:, 3], y_testing[:, 0])
    ax1[1][1].scatter(x_training[:, 3], y_training[:, 0], color='red')
    plt.savefig(path + 'phase_UTS.png', bbox_inches='tight')
    # YS
    fig1, ax1 = plt.subplots(2, 2, figsize=(16, 12))
    # Al_1/YS
    ax1[0][0].set_xlabel('Al_1 / wt.%')
    ax1[0][0].set_ylabel('YS / MPa')
    ax1[0][0].set_title('Al_1/YS', fontstyle='oblique')
    ax1[0][0].scatter(x_testing[:, 0], y_testing[:, 1])
    ax1[0][0].scatter(x_training[:, 0], y_training[:, 1], color='red')
    # Al_2/YS
    ax1[0][1].set_xlabel('Al_2 / wt.%')
    ax1[0][1].set_ylabel('YS / MPa')
    ax1[0][1].set_title('Al_2/YS', fontstyle='oblique')
    ax1[0][1].scatter(x_testing[:, 1], y_testing[:, 1])
    ax1[0][1].scatter(x_training[:, 1], y_training[:, 1], color='red')
    # Si/YS
    ax1[1][0].set_xlabel('Si / wt.%')
    ax1[1][0].set_ylabel('YS / MPa')
    ax1[1][0].set_title('Si/YS', fontstyle='oblique')
    ax1[1][0].scatter(x_testing[:, 2], y_testing[:, 1])
    ax1[1][0].scatter(x_training[:, 2], y_training[:, 1], color='red')
    # AlSc2Si2/YS
    ax1[1][1].set_xlabel('AlSc2Si2 / wt.%')
    ax1[1][1].set_ylabel('YS / MPa')
    ax1[1][1].set_title('AlSc2Si2/YS', fontstyle='oblique')
    ax1[1][1].scatter(x_testing[:, 3], y_testing[:, 1])
    ax1[1][1].scatter(x_training[:, 3], y_training[:, 1], color='red')
    plt.savefig(path + 'phase_YS.png', bbox_inches='tight')
    # EL
    fig1, ax1 = plt.subplots(2, 2, figsize=(16, 12))
    # Al_1/EL
    ax1[0][0].set_xlabel('Al_1 / wt.%')
    ax1[0][0].set_ylabel('EL / MPa')
    ax1[0][0].set_title('Al_1/EL', fontstyle='oblique')
    ax1[0][0].scatter(x_testing[:, 0], y_testing[:, 2])
    ax1[0][0].scatter(x_training[:, 0], y_training[:, 2], color='red')
    # Al_2/EL
    ax1[0][1].set_xlabel('Al_2 / wt.%')
    ax1[0][1].set_ylabel('EL / MPa')
    ax1[0][1].set_title('Al_2/EL', fontstyle='oblique')
    ax1[0][1].scatter(x_testing[:, 1], y_testing[:, 2])
    ax1[0][1].scatter(x_training[:, 1], y_training[:, 2], color='red')
    # Si/EL
    ax1[1][0].set_xlabel('Si / wt.%')
    ax1[1][0].set_ylabel('EL / MPa')
    ax1[1][0].set_title('Si/EL', fontstyle='oblique')
    ax1[1][0].scatter(x_testing[:, 2], y_testing[:, 2])
    ax1[1][0].scatter(x_training[:, 2], y_training[:, 2], color='red')
    # AlSc2Si2/EL
    ax1[1][1].set_xlabel('AlSc2Si2 / wt.%')
    ax1[1][1].set_ylabel('EL / MPa')
    ax1[1][1].set_title('AlSc2Si2/EL', fontstyle='oblique')
    ax1[1][1].scatter(x_testing[:, 3], y_testing[:, 2])
    ax1[1][1].scatter(x_training[:, 3], y_training[:, 2], color='red')
    plt.savefig(path + 'phase_EL.png', bbox_inches='tight')
    plt.show()

# 程序入口
def main():
    # 获取数据
    x, y_UTS, y_YS, y_EL, EL_Si, EL_Mg = get_training_data(
        training_data_file_path)
    x_testing, EL_Si_test, EL_Mg_test = get_testing_data(
        testing_data_file_path)

    # 执行正则化，并记住训练集数据的正则化规则,运用于测试集数据
    x_scaler = StandardScaler().fit(x)
    x_standarded = torch.from_numpy(x_scaler.transform(x)).float()
    # print(x_standarded)
    x_standarded_test = torch.from_numpy(x_scaler.transform(x_testing)).float()

    # 执行模型训练
    y_list = torch.cat((y_UTS, y_YS, y_EL), 1)
    training_break = train(x_standarded, y_list)

    # 调用训练好的模型进行预测
    if not training_break:
        model_path = path + 'model-v1.4.4.pkl'
        # 此处不需要跟踪梯度
        with torch.no_grad():
            y_testing = test(model_path, x_standarded_test)
            if np.isnan(y_testing.numpy().any()):
                print('Prediction run out of range!')
            else:
                print('==================Prediction complete===================')
                print('The index of file ---> %.3f' % index)

                # 数据可视化(散点图)
                draw_scatter(EL_Si.numpy(), EL_Mg.numpy(), y_UTS.numpy(),
                             EL_Si_test.numpy(), EL_Mg_test.numpy(), y_testing.numpy()[:, 0], 'UTS / MPa')
                draw_scatter(EL_Si.numpy(), EL_Mg.numpy(), y_YS.numpy(),
                             EL_Si_test.numpy(), EL_Mg_test.numpy(), y_testing.numpy()[:, 1], 'YS / MPa')
                draw_scatter(EL_Si.numpy(), EL_Mg.numpy(), y_EL.numpy(),
                             EL_Si_test.numpy(), EL_Mg_test.numpy(), y_testing.numpy()[:, 2], 'EL / %')

                # 综合力学性能计算及可视化
                # data_process(path, EL_Si_test.numpy(), EL_Mg_test.numpy())

                # 绘制相分数-性能关系图
                draw_relation(x.numpy(), y_list.numpy(),
                              x_testing.numpy(), y_testing.numpy())


if __name__ == '__main__':
    main()
