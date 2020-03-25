###########################################################
#                net shape: n-10-5-3                      #
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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler


# 程序入口
def main(parameters_list):

    # 初始化参数

    learning_rate = parameters_list[0]
    loss_threashold_value = parameters_list[1]
    error = parameters_list[2]
    loop_max = parameters_list[3]
    path = parameters_list[4]
    train_start_index = parameters_list[5]
    train_end_index = parameters_list[6]
    training_data_file_path = parameters_list[7]
    predicting_data_file_path = parameters_list[8]
    upper_limit = parameters_list[9]
    lower_limit = parameters_list[10]
    Net = parameters_list[11]
    features = parameters_list[12]
    layer_1 = parameters_list[13]
    layer_2 = parameters_list[14]
    old_model_path = parameters_list[15]

    # 定义获取预测数据函数

    def get_predicting_data(file_path):
        data = pd.read_csv(file_path)
        x = torch.from_numpy(data.loc[:, 'PH_Al':'PH_AlSc2Si2'].values).float()
        EL_Si = torch.unsqueeze(
            (torch.from_numpy(data['EL_Si'].values)), dim=1).float()
        EL_Mg = torch.unsqueeze(
            (torch.from_numpy(data['EL_Mg'].values)), dim=1).float()
        return x, EL_Si, EL_Mg

    # 定义获取测试数据函数

    def get_testing_data(file_path):
        data = pd.read_csv(file_path)
        x = torch.from_numpy(data.loc[:, 'PH_Al':'PH_AlSc2Si2'].values).float()
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
        return x, y_UTS, y_YS, y_EL

    # 定义获取训练数据函数

    def get_training_data(file_path):
        data = pd.read_csv(file_path)
        data = data.iloc[train_start_index:train_end_index, :]
        x = torch.from_numpy(data.loc[:, 'PH_Al':'PH_AlSc2Si2'].values).float()
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

    def train(x, y, old_model_path):
        # 实例化神经网络
        net = Net(n_feature=features, n_hidden1=layer_1,
                  n_hidden2=layer_2, n_output=3)
        # 加载模型
        old_model_dict = torch.load(old_model_path).state_dict()
        net.load_state_dict(old_model_dict)
        # Adam优化器
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        # 损失函数
        loss_func = torch.nn.MSELoss()
        # 数据初始化
        loop = 0
        loop_added = 0
        training_break = False
        start_time = time.time()
        predict_y = net(x)
        loss_y = torch.abs(predict_y - y)
        loss = loss_func(loss_y, error)
        # 图像配置初始化
        sns.set(font="Times New Roman")
        fig = plt.figure()
        ax = plt.gca()
        ax.set_title('Learning rate: %.2e' % learning_rate)
        ax.set_xlabel('Loops / K')
        ax.set_ylabel('Loss value')
        ax.set_ylim(lower_limit, upper_limit)
        # 循环训练
        while loss > loss_threashold_value:
            loop += 1
            predict_y = net(x)
            loss_y = torch.abs(predict_y - y)
            loss = loss_func(loss_y, error)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (loop <= loop_max + loop_added):
                if (loop % 1000 == 0):
                    print('Loop: %dK ---' % (loop / 1000),
                          'loss: %.6f' % loss.item())
                    # 可视化误差变化
                    if loss.item() <= upper_limit:
                        ax.scatter(loop / 1000, loss.item(), color='red', s=5)
                        plt.pause(0.1)
            else:
                user_choice = input('Continue or not(Y/N)')
                if (user_choice.lower() != 'y'):
                    training_break = True
                    print('Training break!!!')
                    break
                else:
                    loop_added += loop_max

        if not training_break:
            end_time = time.time()
            if os.path.exists(path):
                pass
            else:
                os.makedirs(path)
            torch.save(net, path + 'model.pkl')
            # w_1 = net.hidden1.weight
            # w_2 = net.hidden2.weight
            # w_3 = net.predict.weight
            # b_1 = net.hidden1.bias
            # b_2 = net.hidden2.bias
            # b_3 = net.predict.bias
            # general_w = (w_3.mm(w_2)).mm(w_1)
            # general_b = w_3.mm(w_2.mm(b_1.view(10, 1))) + \
            #     w_3.mm(b_2.view(5, 1)) + b_3.view(3, 1)
            print('===================Training complete====================')
            print('Total time: %.2fs' % (end_time - start_time))
            # print('Layer1 weight ---> ', w_1)
            # print('Layer1 bias ---> ', b_1)
            # print('Layer2 weight ---> ', w_2)
            # print('Layer2 bias ---> ', b_2)
            # print('Layer3 weight ---> ', w_3)
            # print('Layer3 bias ---> ', b_3)
            # print('General weight calculated from NN ---> \n', general_w)
            # print('General bias calculated from NN ---> \n', general_b)
            # np.savetxt(path + 'w_1.csv',
            #            w_1.detach().numpy(), fmt='%.3f', delimiter=',')
            # np.savetxt(path + 'w_2.csv',
            #            w_2.detach().numpy(), fmt='%.3f', delimiter=',')
            # np.savetxt(path + 'w_3.csv',
            #            w_3.detach().numpy(), fmt='%.3f', delimiter=',')
            # np.savetxt(path + 'b_1.csv',
            #            b_1.detach().numpy(), fmt='%.3f', delimiter=',')
            # np.savetxt(path + 'b_2.csv',
            #            b_2.detach().numpy(), fmt='%.3f', delimiter=',')
            # np.savetxt(path + 'b_3.csv',
            #            b_3.detach().numpy(), fmt='%.3f', delimiter=',')
            # np.savetxt(path + 'general_w.csv',
            #            general_w.detach().numpy(), fmt='%.3f', delimiter=',')
            # np.savetxt(path + 'general_b.csv', general_b.detach().numpy(),
            #            fmt='%.3f', delimiter=',')
            plt.savefig(path + 'learning_curve.png')
            plt.show()

        return training_break

    # 定义测试函数

    def predict(model_path, x):
        net = torch.load(model_path)
        predict_y = net(x)
        pd.DataFrame(predict_y.numpy()).to_csv(
            path + 'predicting_results.csv', index=False, header=['UTS', 'YS', 'EL'])
        return predict_y

    # 绘制散点图

    def draw_scatter(x_training, y_training, z_training, x_predicting, y_predicting, z_predicting, item, training_accuracy, testing_accuracy):
        if item == 'UTS / MPa':
            fig_name = 'UTS'
        elif item == 'YS / MPa':
            fig_name = 'YS'
        else:
            fig_name = 'EL'
        sns.set(font="Times New Roman", font_scale=1)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_title('                    Training accuracy: %.2f %%  Testing accuracy: %.2f %%' % (
            training_accuracy * 100, testing_accuracy * 100))
        ax.set_xlabel('Si')
        ax.set_ylabel('Mg')
        ax.set_zlabel(item)
        ax.scatter(x_training, y_training, z_training,
                   color='red', s=50, label='Training data')
        ax.scatter(x_predicting, y_predicting,
                   z_predicting, label='Predicting data')
        ax.legend(loc='upper left')
        plt.savefig(path + 'elements_%s.png' % fig_name)
        plt.show()

    # 获取数据

    x, y_UTS, y_YS, y_EL, EL_Si, EL_Mg = get_training_data(
        training_data_file_path)
    x_testing, y_UTS_testing, y_YS_testing, y_EL_testing = get_testing_data(
        training_data_file_path)
    x_predicting, EL_Si_predicting, EL_Mg_predicting = get_predicting_data(
        predicting_data_file_path)

    # 执行正则化，并记住训练集数据的正则化规则,运用于测试集数据

    x_scaler = StandardScaler().fit(x)
    x_standarded = torch.from_numpy(x_scaler.transform(x)).float()
    x_standarded_test = torch.from_numpy(
        x_scaler.transform(x_testing)).float()
    x_standarded_predict = torch.from_numpy(
        x_scaler.transform(x_predicting)).float()

    # 执行模型训练

    y_list = torch.cat((y_UTS, y_YS, y_EL), 1)
    y_testing_list = torch.cat((y_UTS_testing, y_YS_testing, y_EL_testing), 1)
    training_break = train(x_standarded, y_list, old_model_path)

    # 调用训练好的模型进行预测

    if not training_break:
        model_path = path + 'model.pkl'
        # 此处不需要跟踪梯度
        with torch.no_grad():
            # 评估
            y_testing = predict(model_path, x_standarded_test)
            y_training = predict(model_path, x_standarded)
            accuracy_func = torch.nn.L1Loss()
            training_accuracy_UTS = 1 - \
                (accuracy_func(
                    y_training[:, 0], y_list[:, 0]) / torch.mean(y_list[:, 0])).item()
            training_accuracy_YS = 1 - \
                (accuracy_func(
                    y_training[:, 1], y_list[:, 1]) / torch.mean(y_list[:, 1])).item()
            training_accuracy_EL = 1 - \
                (accuracy_func(
                    y_training[:, 2], y_list[:, 2]) / torch.mean(y_list[:, 2])).item()
            training_accuracy = [training_accuracy_UTS,
                                 training_accuracy_YS, training_accuracy_EL]
            testing_accuracy_UTS = 1 - \
                (accuracy_func(
                    y_testing[:, 0], y_testing_list[:, 0]) / torch.mean(y_testing_list[:, 0])).item()
            testing_accuracy_YS = 1 - \
                (accuracy_func(
                    y_testing[:, 1], y_testing_list[:, 1]) / torch.mean(y_testing_list[:, 1])).item()
            testing_accuracy_EL = 1 - \
                (accuracy_func(
                    y_testing[:, 2], y_testing_list[:, 2]) / torch.mean(y_testing_list[:, 2])).item()
            testing_accuracy = [testing_accuracy_UTS,
                                testing_accuracy_YS, testing_accuracy_EL]
            # 预测
            y_predicting = predict(model_path, x_standarded_predict)
            if np.isnan(y_predicting.numpy().any()):
                print('==============Prediction run out of range===============')
            else:
                print('==================Prediction complete===================')

                # 数据可视化(散点图)
                draw_scatter(EL_Si.numpy(), EL_Mg.numpy(), y_UTS.numpy(),
                             EL_Si_predicting.numpy(), EL_Mg_predicting.numpy(), y_predicting.numpy()[:, 0], 'UTS / MPa', training_accuracy[0], testing_accuracy[0])
                draw_scatter(EL_Si.numpy(), EL_Mg.numpy(), y_YS.numpy(),
                             EL_Si_predicting.numpy(), EL_Mg_predicting.numpy(), y_predicting.numpy()[:, 1], 'YS / MPa', training_accuracy[1], testing_accuracy[1])
                draw_scatter(EL_Si.numpy(), EL_Mg.numpy(), y_EL.numpy(),
                             EL_Si_predicting.numpy(), EL_Mg_predicting.numpy(), y_predicting.numpy()[:, 2], 'EL / %', training_accuracy[2], testing_accuracy[2])
    return training_break


if __name__ == '__main__':
    main(parameters_list)
