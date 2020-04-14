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
    EL_Sr_predict = parameters_list[3]
    ANN_II_layer_1 = parameters_list[4]
    Net = parameters_list[5]
    path = parameters_list[6]
    old_model_path = parameters_list[7]
    learning_rate = parameters_list[8]
    loss_threashold_value = parameters_list[9]
    train_start_index = parameters_list[10]
    train_end_index = parameters_list[11]
    error = parameters_list[12]
    add = parameters_list[13]

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

    def draw_scatter(x_training, y_training, x_predicting, y_predicting, types, error, add, item=None):
        sns.set(font="Times New Roman", font_scale=1.3, style='ticks')
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'

        if add:
            y_index = -1
        else:
            y_index = train_end_index

        if types == 'whole':
            fig = plt.figure(figsize=(8, 6))
            ax = brokenaxes.brokenaxes(
                ylims=((-0.03, 0.09), (0.39, 1.25)), hspace=0.05, despine=False)
            ax.set_xlabel('Sr / wt. %')
            ax.set_ylabel('Phase fraction / wt. %')
            # Predicted
            ax.scatter(x_predicting, y_predicting[:, 0],
                       s=15, color='cornflowerblue',
                       label='Predicted Al phase')
            ax.scatter(x_predicting, y_predicting[:, 1],
                       s=15, color='chocolate',
                       label='Predicted Si phase')
            ax.scatter(x_predicting, y_predicting[:, 2],
                       s=15, color='mediumseagreen',
                       label='Predicted Al${_2}$Si${_2}$Sr phase')
            # Additional label
            if add:
                # Al2Si2Sr
                x_add3 = np.array([x_training[y_index]])
                y_add3 = np.array([y_training[y_index, 2]])
                e_add3 = np.array([[0.01],
                                   [0.01]])
                ax.errorbar(x_add3, y_add3, yerr=e_add3,
                            linestyle='None', capsize=5, ecolor='r',
                            fmt='*', mfc='white', mec='r',
                            label='Additional experiment', ms=10)
            # Training
            ax.errorbar(x_training[0:y_index], y_training[0:y_index, 0], yerr=error[0],
                        linestyle='None', capsize=5, ecolor='royalblue',
                        fmt='o:', mfc='wheat', mec='royalblue',
                        label='Experimental Al phase')
            ax.errorbar(x_training[0:y_index], y_training[0:y_index, 1], yerr=error[1],
                        linestyle='None', capsize=5, ecolor='saddlebrown',
                        fmt='o:', mfc='wheat', mec='saddlebrown',
                        label='Experimental Si phase')
            ax.errorbar(x_training[0:y_index], y_training[0:y_index, 2], yerr=error[2],
                        linestyle='None', capsize=5, ecolor='green',
                        fmt='o:', mfc='wheat', mec='green',
                        label='Experimental Al${_2}$Si${_2}$Sr phase')
            # Additional
            if add:
                # Al
                x_add1 = np.array([x_training[y_index]])
                y_add1 = np.array([y_training[y_index, 0]])
                e_add1 = np.array([[0.03],
                                   [0.03]])
                ax.errorbar(x_add1, y_add1, yerr=e_add1,
                            linestyle='None', capsize=5, ecolor='r',
                            fmt='*', mfc='white', mec='r', ms=10)
                # Si
                x_add2 = np.array([x_training[y_index]])
                y_add2 = np.array([y_training[y_index, 1]])
                e_add2 = np.array([[0.02],
                                   [0.02]])
                ax.errorbar(x_add2, y_add2, yerr=e_add2,
                            linestyle='None', capsize=5, ecolor='r',
                            fmt='*', mfc='white', mec='r', ms=10)
                # Al2Si2Sr
                x_add3 = np.array([x_training[y_index]])
                y_add3 = np.array([y_training[y_index, 2]])
                e_add3 = np.array([[0.01],
                                   [0.01]])
                ax.errorbar(x_add3, y_add3, yerr=e_add3,
                            linestyle='None', capsize=5, ecolor='r',
                            fmt='*', mfc='white', mec='r', ms=10)
            ax.legend(loc='upper right', frameon=False, ncol=2)
            plt.savefig(path + 'elements_%s.png' % types)
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.subplot()
            ax.set_xlabel('Sr / wt. %')
            if item == 'Al / wt. %':
                item_ = 'Phase fraction of Al phase / wt. %'
                fig_name = 'Al'
                index = 0
                y_min = 0.4
                y_max = 1
                ax.vlines(0.077, 0, 1)
            elif item == 'Si / wt. %':
                item_ = 'Phase fraction of Si phase / wt. %'
                fig_name = 'Si'
                index = 1
                y_min = 0
                y_max = 0.1
            else:
                item_ = 'Phase fraction of Al${_2}$Si${_2}$Sr phase/ wt. %'
                fig_name = 'AlSi2Sr'
                index = 2
                y_min = -0.1
                y_max = 0.1
            ax.set_ylabel(item_)
            ax.set_ylim(y_min, y_max)
            ax.scatter(x_predicting, y_predicting[:, index],
                       label='Predicted data')
            # ax.errorbar(x_training, y_training[:, index], yerr=error[index],
            #             linestyle='None', capsize=5, ecolor='saddlebrown',
            #             fmt='o:', mfc='wheat', mec='saddlebrown',
            #             label='Experimental Si phase')
            ax.scatter(x_training, y_training[:, index],
                       s=50, label='Experimental data')
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

                draw_scatter(EL_Sr.numpy(), y.numpy(), EL_Sr_predict.numpy(),
                             y_predicting.numpy(), 'part', error, add, item='Al / wt. %',)
                draw_scatter(EL_Sr.numpy(), y.numpy(), EL_Sr_predict.numpy(),
                             y_predicting.numpy(), 'part', error, add, item='Si / wt. %', )
                draw_scatter(EL_Sr.numpy(), y.numpy(), EL_Sr_predict.numpy(),
                             y_predicting.numpy(), 'part', error, add, item='AlSi2Sr / wt. %', )
                draw_scatter(EL_Sr.numpy(), y.numpy(), EL_Sr_predict.numpy(),
                             y_predicting.numpy(), 'whole', error, add)

    return training_break


if __name__ == '__main__':
    main(parameters_list)