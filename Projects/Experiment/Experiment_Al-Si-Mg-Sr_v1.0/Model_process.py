import os

import matplotlib
import matplotlib.pyplot as plt
import brokenaxes
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def main(parameters_list):

    # 初始化参数

    training_data_file_path = parameters_list[0]
    predicting_data_file_path = parameters_list[1]
    path = parameters_list[2]
    model_path = parameters_list[3]
    train_start_index = parameters_list[4]
    train_end_index = parameters_list[5]
    error = parameters_list[6]
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

    # 定义获取预测数据函数

    def get_predicting_data(file_path):
        data = pd.read_csv(file_path)
        x = torch.from_numpy(data.loc[:, 'PH_Al':'PH_AlSi2Sr'].values).float()
        EL_Sr = torch.unsqueeze(
            (torch.from_numpy(data['EL_Sr'].values)), dim=1).float()
        return x, EL_Sr

    # 定义获取测试数据函数

    def get_testing_data(file_path):
        data = pd.read_csv(file_path)
        x = torch.from_numpy(data.loc[:, 'PH_Al':'PH_AlSi2Sr'].values).float()
        y_UTS = torch.unsqueeze(
            (torch.from_numpy(data['UTS'].values)), dim=1).float()
        y_YS = torch.unsqueeze(
            (torch.from_numpy(data['YS'].values)), dim=1).float()
        y_EL = torch.unsqueeze(
            (torch.from_numpy(data['EL'].values)), dim=1).float()
        return x, y_UTS, y_YS, y_EL

    # 定义获取训练数据函数

    def get_training_data(file_path):
        data = pd.read_csv(file_path)
        data = data.iloc[train_start_index:train_end_index, :]
        x = torch.from_numpy(data.loc[:, 'PH_Al':'PH_AlSi2Sr'].values).float()
        y_UTS = torch.unsqueeze(
            (torch.from_numpy(data['UTS'].values)), dim=1).float()
        y_YS = torch.unsqueeze(
            (torch.from_numpy(data['YS'].values)), dim=1).float()
        y_EL = torch.unsqueeze(
            (torch.from_numpy(data['EL'].values)), dim=1).float()
        EL_Sr = torch.unsqueeze(
            (torch.from_numpy(data['EL_Sr'].values)), dim=1).float()
        return x, y_UTS, y_YS, y_EL, EL_Sr

    # 定义测试函数

    def predict(model_path, x):
        net = torch.load(model_path)
        predict_y = net(x)
        pd.DataFrame(predict_y.numpy()).to_csv(
            path + 'predicting_results.csv', index=False, header=['UTS', 'YS', 'EL'])
        return predict_y

    # 综合处理全部数据

    def data_process(path, x, y):
        data = pd.read_csv(path + 'predicting_results.csv')
        mms = MinMaxScaler()
        data_processed = mms.fit_transform(data.values)
        data_calculated = 0.5 * \
            data_processed[:, 0] + 0.5 * \
            data_processed[:, 1] + data_processed[:, 2]
        # data_calculated = 0.5 * data_processed[:, 0] + 0.5 * data_processed[:, 1]
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
                   data_calculated[max_index], color='red', s=50, label='Optimal general performance\nSi: %.2fwt. %%  Mg: %.2fwt. %%' % (x[max_index][0], y[max_index][0]))
        ax.legend(loc='upper right', frameon=False)
        plt.savefig(path + 'general_Performance.png')
        # plt.show()

    # 绘制散点图

    def draw_scatter(x_training, y_training, x_predicting, y_predicting, error):
        # 7data
        ymin1 = 85
        ymax1 = 240
        ymin2 = 7
        ymax2 = 18
        # 6data
        ymin1 = 80
        ymax1 = 250
        ymax2 = 20

        sns.set(font="Times New Roman", font_scale=1.3, style='ticks')
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'
        fig = plt.figure(figsize=(8, 6))
        ax1 = plt.subplot()
        ax2 = ax1.twinx()
        ax1.set_xlabel('Sr / wt. %')
        ax1.set_ylabel('Strength / MPa')
        ax1.set_ylim(ymin1, ymax1)
        ax2.set_ylabel('Elongation / %', color='chocolate')
        ax2.spines['right'].set_color('chocolate')
        ax2.yaxis.label.set_color('chocolate')
        ax2.tick_params(axis='y', colors='chocolate')
        ax2.set_ylim(ymin2, ymax2)

        # UTS
        pl11 = ax1.scatter(x_predicting, y_predicting[:, 0],
                           s=15, color='cornflowerblue')
        ax1.errorbar(x_training, y_training[:, 0], yerr=error[0], linestyle='None',
                     capsize=5, ecolor='royalblue')
        pl12 = ax1.scatter(
            x_training, y_training[:, 0], s=25, color='royalblue')
        # YS
        pl21 = ax1.scatter(x_predicting, y_predicting[:, 1],
                           s=15, color='mediumseagreen')
        ax1.errorbar(x_training, y_training[:, 1], yerr=error[1], linestyle='None',
                     capsize=5, ecolor='green')
        pl22 = ax1.scatter(x_training, y_training[:, 1], s=25, color='green')
        # EL
        pl31 = ax2.scatter(x_predicting, y_predicting[:, 2],
                           s=15, color='chocolate')
        ax2.errorbar(x_training, y_training[:, 2], yerr=error[2], linestyle='None',
                     capsize=5, ecolor='saddlebrown')
        pl32 = ax2.scatter(
            x_training, y_training[:, 2], s=25, color='saddlebrown')

        ax1.vlines(0.005, ymin1, ymax1, linestyles='dashed',
                   color='r', linewidth=2)
        ax1.vlines(0.075, ymin1, ymax1, linestyles='dashed',
                   color='r', linewidth=2)
        ax1.text(0.01, 130, 'w(Sr) = 0.005', color='r')
        ax1.text(0.08, 130, 'w(Sr) = 0.075', color='r')

        label11 = 'Predicted UTS'
        label12 = 'Experimental UTS'
        label21 = 'Predicted YS'
        label22 = 'Experimental YS'
        label31 = 'Predicted EL'
        label32 = 'Experimental EL'

        plt.legend([pl11, pl21, pl31, pl12, pl22, pl32],
                   [label11, label21, label31, label12, label22, label32],
                   loc='upper right', frameon=False, ncol=2)
        plt.savefig(path + 'performance.png')

        # plt.show()

    # 线性拟合函数

    def linear_fitting(x, y):
        parameters = np.polyfit(x, y, 1)
        return parameters[0], parameters[1]

    # 绘制相分数(归一化)-性能(归一化)关系图

    def draw_relation_allRE(x_training_, y_training_, x_testing_, y_testing_, item):
        sns.set(font="Times New Roman", font_scale=1.3, style='ticks')
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'
        y_min = -7
        y_max = 5

        if item == 'Al':
            item_name = r'$\alpha$-(Al)'
            x_phase = np.linspace(-1.5, 1, 100)
            # x_index1 = -1
            x_index1 = 151
            x_index2 = 0
        elif item == 'Si':
            item_name = 'Eutectic (Si)'
            x_phase = np.linspace(-1, 2.5, 100)
            x_index1 = 151
            x_index2 = 1
            y_max = 7
        else:
            item_name = 'Al${_2}$Si${_2}$Sr'
            x_phase = np.linspace(-1, 2, 100)
            x_index1 = 151
            x_index2 = 2
            y_max = 7

        xlabel = 'Phase fraction of %s with regularization' % item_name
        ylabel = 'Performance with regularization'

        fig = plt.figure(figsize=(8, 6))
        ax = plt.subplot()
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # UTS
        pl11 = ax.scatter(x_testing_[:, x_index2], y_testing_[:, 0], s=15)
        fitting_w1, fitting_b1 = linear_fitting(x_testing_[0:x_index1, x_index2],
                                                y_testing_[0:x_index1, 0])
        pl12, = ax.plot(x_phase, fitting_w1 * x_phase +
                        fitting_b1, linestyle='dashed', linewidth=2)
        # YS
        pl21 = ax.scatter(x_testing_[:, x_index2], y_testing_[:, 1], s=15)
        fitting_w2, fitting_b2 = linear_fitting(x_testing_[0:x_index1, x_index2],
                                                y_testing_[0:x_index1, 1])
        pl22, = ax.plot(x_phase, fitting_w2 * x_phase +
                        fitting_b2, linestyle='dashed', linewidth=2)
        # EL
        pl31 = ax.scatter(x_testing_[:, x_index2], y_testing_[:, 2], s=15)
        fitting_w3, fitting_b3 = linear_fitting(x_testing_[0:x_index1, x_index2],
                                                y_testing_[0:x_index1, 2])
        pl32, = ax.plot(x_phase, fitting_w3 * x_phase +
                        fitting_b3, linestyle='dashed', linewidth=2)

        label11 = 'UTS'
        label12 = 'w${_{UTS}}$ = %.2f  b${_{UTS}}$ = %.2f' % (fitting_w1, fitting_b1)
        label21 = 'YS'
        label22 = 'w${_{YS}}$ = %.2f  b${_{YS}}$ = %.2f' % (fitting_w2, fitting_b2)
        label31 = 'EL'
        label32 = 'w${_{EL}}$ = %.2f  b${_{EL}}$ = %.2f' % (fitting_w3, fitting_b3)

        plt.legend([pl11, pl21, pl31, pl12, pl22, pl32],
                   [label11, label21, label31, label12, label22, label32],
                   loc='upper left', frameon=False, ncol=2)
        plt.savefig(path + '%s_performance_allRE.png' % item)
        linear_coef = pd.DataFrame(data=np.ones((3, 2)),
                                         index=['UTS', 'YS', 'EL'],
                                         columns=['weight', 'bias'])
        linear_coef.iloc[0, 0] = fitting_w1
        linear_coef.iloc[0, 1] = fitting_b1
        linear_coef.iloc[1, 0] = fitting_w2
        linear_coef.iloc[1, 1] = fitting_b2
        linear_coef.iloc[2, 0] = fitting_w3
        linear_coef.iloc[2, 1] = fitting_b3

        linear_coef.to_csv(
            path + '%s_linear_coef_allRE.csv' % item, float_format='%.2f')
        # plt.show()

    # 获取数据

    x, y_UTS, y_YS, y_EL, EL_Sr = get_training_data(
        training_data_file_path)
    x_testing, y_UTS_testing, y_YS_testing, y_EL_testing = get_testing_data(
        training_data_file_path)
    x_predicting, EL_Sr_predicting = get_predicting_data(
        predicting_data_file_path)

    # 执行正则化，并记住训练集数据的正则化规则,运用于测试集数据

    x_scaler = StandardScaler().fit(x)
    x_standarded = torch.from_numpy(x_scaler.transform(x)).float()
    x_standarded_test = torch.from_numpy(
        x_scaler.transform(x_testing)).float()
    x_standarded_predict = torch.from_numpy(
        x_scaler.transform(x_predicting)).float()

    # 调用模型进行预测

    y_list = torch.cat((y_UTS, y_YS, y_EL), 1)
    y_testing_list = torch.cat((y_UTS_testing, y_YS_testing, y_EL_testing), 1)
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

        y_scaler = StandardScaler().fit(y_predicting)
        y_standarded = torch.from_numpy(y_scaler.transform(y_list)).float()
        y_standarded_predict = torch.from_numpy(
            y_scaler.transform(y_predicting)).float()
        # print(y_standarded_predict[:, 0].min())
        # print(y_standarded_predict[:, 0].max())
        # print(y_standarded_predict[:, 1].min())
        # print(y_standarded_predict[:, 1].max())
        # print(y_standarded_predict[:, 2].min())
        # print(y_standarded_predict[:, 2].max())
        # print(y_predicting[:, 0].min())
        # print(y_predicting[:, 0].max())
        # print(y_predicting[:, 1].min())
        # print(y_predicting[:, 1].max())
        # print(y_predicting[:, 2].min())
        # print(y_predicting[:, 2].max())
        if np.isnan(y_predicting.numpy().any()):
            print('==============Prediction run out of range===============')
        else:
            print('==================Prediction complete===================')

            # 数据可视化(散点图)
            draw_scatter(EL_Sr.numpy(), y_list.numpy(),
                         EL_Sr_predicting.numpy(), y_predicting.numpy(), error)

            # 综合力学性能计算及可视化
            # data_process(path, EL_Si_predicting.numpy(), EL_Mg_predicting.numpy())

            # 绘制相分数(正则化)-性能(正则化)关系图
            draw_relation_allRE(x_standarded.numpy(), y_standarded.numpy(),
                                x_standarded_predict.numpy(), y_standarded_predict.numpy(),
                                'Al')
            draw_relation_allRE(x_standarded.numpy(), y_standarded.numpy(),
                                x_standarded_predict.numpy(), y_standarded_predict.numpy(),
                                'Si')
            draw_relation_allRE(x_standarded.numpy(), y_standarded.numpy(),
                                x_standarded_predict.numpy(), y_standarded_predict.numpy(),
                                'Al2Si2Sr')


if __name__ == '__main__':
    main(parameters_list)
