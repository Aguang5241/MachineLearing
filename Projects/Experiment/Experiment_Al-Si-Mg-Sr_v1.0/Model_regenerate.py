import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


def main(parameters_list):

    # 初始化参数

    training_data_file_path = parameters_list[0]
    features = parameters_list[1]
    loop_max = parameters_list[2]
    EL_Sr_predict = parameters_list[3]
    ANN_II_layer_1 = parameters_list[4]
    Net = parameters_list[5]
    path = parameters_list[6]
    old_model_path = parameters_list[7]
    learning_rate = parameters_list[8]
    loss_threashold_value = parameters_list[9]
    train_start_index = parameters_list[10]
    train_end_index = parameters_list[11]

    # 获取训练数据

    def get_training_data(file_path):
        data = pd.read_csv(file_path)
        data = data.iloc[train_start_index:train_end_index, :]
        EL_Sr = torch.unsqueeze(
            (torch.from_numpy(data['EL_Sr'].values)), dim=1).float()
        y = torch.from_numpy(data.loc[:, 'PH_Al':'PH_AlSi2Sr'].values).float()
        return EL_Sr, y

    # 定义训练函数

    def train(x, y, old_model_path):
        # 实例化神经网络
        net = Net(n_feature=1, n_hidden=ANN_II_layer_1,
                  n_output=features)
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
        loss = 1e9
        # 循环训练
        while loss > loss_threashold_value:
            loop += 1
            predict_y = net(x)
            loss = loss_func(predict_y, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (loop <= loop_max + loop_added):
                if (loop % 1000 == 0):
                    print('Loop: %dK ---' % (loop / 1000),
                          'loss: %.9f' % loss.item())
            else:
                user_choice = input('Continue or not(Y/N)')
                if (user_choice.lower() != 'y'):
                    training_break = True
                    print('Training break!!!')
                    break
                else:
                    loop_added += loop_max

        if not training_break:
            if os.path.exists(path):
                pass
            else:
                os.makedirs(path)
            torch.save(net, path + 'generate_model.pkl')
            print('===================Training complete====================')

        return training_break

    # 定义预测函数

    def predict(model_path, x_standarded, x):
        net = torch.load(model_path)
        predict_y = net(x_standarded)
        results = torch.cat((predict_y, x), 1)
        pd.DataFrame(results.numpy()).to_csv(
            path + 'generate_results.csv',
            header=['PH_Al', 'PH_Si', 'PH_AlSi2Sr', 'EL_Sr'])
        return predict_y

    # 绘制散点图

    def draw_scatter(x_training, y_training, x_predicting, y_predicting, item):
        if item == 'Al / wt. %':
            item_ = r'$\alpha$-(Al) / wt. %'
            fig_name = 'Al'
        elif item == 'Si / wt. %':
            item_ = 'Eutectic (Si) / wt. %'
            fig_name = 'Si'
        else:
            item_ = 'A${l_2}$S${i_2}$Sr / wt. %'
            fig_name = 'AlSi2Sr'
        sns.set(font="Times New Roman", font_scale=1.3, style='ticks')
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'
        fig = plt.figure(figsize=(8, 6))
        ax = plt.subplot()
        ax.set_xlabel('Sr / wt. %')
        ax.set_ylabel(item_)
        ax.scatter(x_predicting, y_predicting, label='Predicting data')
        ax.scatter(x_training, y_training, s=50, label='Training data')
        ax.legend(loc='upper right', frameon=False)
        plt.savefig(path + 'elements_%s.png' % fig_name)
        plt.show()

    # 获取数据

    EL_Sr, y = get_training_data(training_data_file_path)

    # 正则化

    x_scaler = StandardScaler().fit(EL_Sr)
    x_standarded = torch.from_numpy(x_scaler.transform(EL_Sr)).float()
    x_standarded_predict = torch.from_numpy(
        x_scaler.transform(EL_Sr_predict)).float()

    # 执行模型训练

    training_break = train(x_standarded, y, old_model_path)

    # 调用训练好的模型进行评估与预测

    if not training_break:
        model_path = path + 'generate_model.pkl'
        # 此处不需要跟踪梯度
        with torch.no_grad():
            # 预测
            y_predicting = predict(
                model_path, x_standarded_predict, EL_Sr_predict)
            if np.isnan(y_predicting.numpy().any()):
                print('==============Prediction run out of range===============')
            else:
                print('==================Prediction complete===================')

                # 数据可视化(散点图)

                draw_scatter(EL_Sr.numpy(), y[:, 0].numpy(), EL_Sr_predict.numpy(),
                             y_predicting[:, 0].numpy(), 'Al / wt. %')
                draw_scatter(EL_Sr.numpy(), y[:, 1].numpy(), EL_Sr_predict.numpy(),
                             y_predicting[:, 1].numpy(), 'Si / wt. %')
                draw_scatter(EL_Sr.numpy(), y[:, 2].numpy(), EL_Sr_predict.numpy(),
                             y_predicting[:, 2].numpy(), 'AlSi2Sr / wt. %')

    return training_break


if __name__ == '__main__':
    main(parameters_list)
