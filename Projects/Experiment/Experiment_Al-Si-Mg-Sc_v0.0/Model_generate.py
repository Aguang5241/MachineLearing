import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib
import brokenaxes
import matplotlib.pyplot as plt


def main(parameters_list):

    # 初始化参数

    training_data_file_path = parameters_list[0]
    features = parameters_list[1]
    loop_max = parameters_list[2]
    EL_Sc_predict = parameters_list[3]
    ANN_II_layer_1 = parameters_list[4]
    Net = parameters_list[5]
    path = parameters_list[6]
    learning_rate = parameters_list[7]
    loss_threashold_value = parameters_list[8]
    train_start_index = parameters_list[9]
    train_end_index = parameters_list[10]

    # 获取训练数据

    def get_training_data(file_path):
        data = pd.read_csv(file_path)
        data = data.iloc[train_start_index:train_end_index, :]
        EL_Sc = torch.unsqueeze(
            (torch.from_numpy(data['EL_Sc'].values)), dim=1).float()
        y = torch.from_numpy(data.loc[:, 'PH_Al':'PH_AlSc2Si2'].values).float()
        return EL_Sc, y

    # 定义训练函数

    def train(x, y):
        # 实例化神经网络
        net = Net(n_feature=1, n_hidden=ANN_II_layer_1,
                  n_output=features)
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
            header=['PH_Al', 'PH_Al_2', 'PH_Si', 'PH_AlSc2Si2', 'EL_Sc'])
        return predict_y

    # 绘制散点图

    def draw_scatter(x_training, y_training, x_predicting, y_predicting,  types, item=None):
        sns.set(font="Times New Roman", font_scale=1.3, style='ticks')
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'
        if types == 'whole':
            fig = plt.figure(figsize=(8, 6))
            ax = plt.subplot()
            ax.set_xlabel('Sc / wt. %')
            ax.set_ylabel('Phase fraction / wt. %')
            # Predicted
            ax.scatter(x_predicting, y_predicting[:, 0],
                       s=15, color='cornflowerblue',
                       label=r'Predicted $\alpha$-(Al) phase')
            ax.scatter(x_predicting, y_predicting[:, 1],
                       s=15, color='chocolate',
                       label='Predicted eutectic (Al) phase')
            ax.scatter(x_predicting, y_predicting[:, 2],
                       s=15, color='mediumseagreen',
                       label='Predicted Si phase')
            ax.scatter(x_predicting, y_predicting[:, 3],
                       s=15, color='violet',
                       label='Predicted AlSc${_2}$Si${_2}$ phase')
            # Training
            ax.scatter(x_training, y_training[:, 0],
                       s=50, color='royalblue',
                       label=r'Experimental $\alpha$-(Al) phase')
            ax.scatter(x_training, y_training[:, 1],
                       s=50, color='saddlebrown',
                       label='Experimental eutectic (Al) phase')
            ax.scatter(x_training, y_training[:, 2],
                       s=50, color='green',
                       label='Experimental Si phase')
            ax.scatter(x_training, y_training[:, 3],
                       s=50, color='magenta',
                       label='Experimental AlSc${_2}$Si${_2}$ phase')

            ax.legend(loc='upper right', frameon=False, ncol=2)

            plt.savefig(path + 'elements_%s.png' % types)
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.subplot()
            if item == 'Al':
                item_ = 'Phase fraction of Al phase / wt. %'
                fig_name = 'Al'
                index = 0
                ax.vlines(0.54, 0, 1)
            elif item == 'Al2':
                item_ = 'Phase fraction of Al2 phase / wt. %'
                fig_name = 'Al2'
                index = 1
                ax.vlines(0.54, 0, 1)
            elif item == 'Si':
                item_ = 'Phase fraction of Si phase / wt. %'
                fig_name = 'Si'
                index = 2
            else:
                item_ = 'Phase fraction of AlSc${_2}$Si${_2}$ phase/ wt. %'
                fig_name = 'AlSc2Si2'
                index = 3
            ax.set_xlabel('Sc / wt. %')
            ax.set_ylabel(item_)
            ax.scatter(
                x_predicting, y_predicting[:, index], label='Predicted data')
            ax.scatter(
                x_training, y_training[:, index], s=50, label='Experimental data')
            ax.legend(loc='upper right', frameon=False)
            plt.savefig(path + 'elements_%s.png' % fig_name)
        plt.show()

    # 获取数据

    EL_Sc, y = get_training_data(training_data_file_path)

    # 正则化

    x_scaler = StandardScaler().fit(EL_Sc)
    x_standarded = torch.from_numpy(x_scaler.transform(EL_Sc)).float()
    x_standarded_predict = torch.from_numpy(
        x_scaler.transform(EL_Sc_predict)).float()

    # 执行模型训练

    training_break = train(x_standarded, y)

    # 调用训练好的模型进行评估与预测

    if not training_break:
        model_path = path + 'generate_model.pkl'
        # 此处不需要跟踪梯度
        with torch.no_grad():
            # 预测
            y_predicting = predict(
                model_path, x_standarded_predict, EL_Sc_predict)
            if np.isnan(y_predicting.numpy().any()):
                print('==============Prediction run out of range===============')
            else:
                print('==================Prediction complete===================')

                # 数据可视化(散点图)

                draw_scatter(EL_Sc.numpy(), y.numpy(), EL_Sc_predict.numpy(),
                             y_predicting.numpy(), 'part', item='Al')
                draw_scatter(EL_Sc.numpy(), y.numpy(), EL_Sc_predict.numpy(),
                             y_predicting.numpy(), 'part', item='Al2')
                draw_scatter(EL_Sc.numpy(), y.numpy(), EL_Sc_predict.numpy(),
                             y_predicting.numpy(), 'part', item='Si')
                draw_scatter(EL_Sc.numpy(), y.numpy(), EL_Sc_predict.numpy(),
                             y_predicting.numpy(), 'part', item='AlSc2Si2')
                draw_scatter(EL_Sc.numpy(), y.numpy(), EL_Sc_predict.numpy(),
                             y_predicting.numpy(), 'whole')

    return training_break


if __name__ == '__main__':
    main(parameters_list)
