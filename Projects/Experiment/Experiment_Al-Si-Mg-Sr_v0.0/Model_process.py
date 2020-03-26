import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def main(parameters_list):

    # 初始化参数

    path = parameters_list[0]
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
    train_start_index = parameters_list[1]
    train_end_index = parameters_list[2]
    training_data_file_path = parameters_list[3]
    predicting_data_file_path = parameters_list[4]
    model_path = parameters_list[5]

    # 定义获取预测数据函数

    def get_predicting_data(file_path):
        data = pd.read_csv(file_path)
        x = torch.from_numpy(data.loc[:, 'PH_Al':'PH_AlSi2Sr'].values).float()
        EL_Sr = torch.unsqueeze(
            (torch.from_numpy(data['EL_Si'].values)), dim=1).float()
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
            (torch.from_numpy(data['EL_Si'].values)), dim=1).float()
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
        ax.legend(loc='upper left')
        plt.savefig(path + 'general_Performance.png')
        # plt.show()

    # 绘制散点图

    def draw_scatter(x_training, y_training, x_predicting, y_predicting, item, training_accuracy, testing_accuracy):
        if item == 'UTS / MPa':
            fig_name = 'UTS'
        elif item == 'YS / MPa':
            fig_name = 'YS'
        else:
            fig_name = 'EL'
        sns.set(font="Times New Roman", font_scale=1)
        fig = plt.figure()
        ax = plt.subplot()
        ax.set_title('Training accuracy: %.2f %%  Testing accuracy: %.2f %%' % (
            training_accuracy * 100, testing_accuracy * 100))
        ax.set_xlabel('Sr / MPa')
        ax.set_ylabel(item)
        ax.scatter(x_predicting, y_predicting, label='Predicting data')
        ax.scatter(x_training, y_training,
                   color='red', s=50, label='Training data')
        ax.legend(loc='upper right')
        plt.savefig(path + 'elements_%s.png' % fig_name)
        # plt.show()

    # 线性拟合函数

    def linear_fitting(x, y):
        parameters = np.polyfit(x, y, 1)
        return parameters[0], parameters[1]

    # 绘制相分数-性能关系图

    def draw_relation(x_training, y_training, x_testing, y_testing):
        sns.set(font="Times New Roman", font_scale=1)
        x_Al_1 = np.linspace(0.305, 0.508, 100)
        x_Al_2 = np.linspace(0, 0.401, 100)
        x_Si = np.linspace(0.046, 0.053, 100)
        x_AlSi2Sr = np.linspace(0, 0.0023, 100)
        # UTS
        fig1, ax1 = plt.subplots(2, 2, figsize=(16, 12))
        # Al_1/UTS
        ax1[0][0].set_xlabel('Al_1 / wt.%')
        ax1[0][0].set_ylabel('UTS / MPa')
        ax1[0][0].set_title('Al_1 / UTS', fontstyle='oblique')
        ax1[0][0].scatter(x_testing[:, 0], y_testing[:, 0],
                          label='Testing data')
        ax1[0][0].scatter(x_training[:, 0], y_training[:, 0],
                          label='Training data')
        fitting_w_100, fitting_b_100 = linear_fitting(x_testing[:, 0],
                                                      y_testing[:, 0])
        ax1[0][0].plot(x_Al_1, fitting_w_100 * x_Al_1 + fitting_b_100, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_100, fitting_w_100))
        ax1[0][0].legend()
        # Al_2/UTS
        ax1[0][1].set_xlabel('Al_2 / wt.%')
        ax1[0][1].set_ylabel('UTS / MPa')
        ax1[0][1].set_title('Al_2 / UTS', fontstyle='oblique')
        ax1[0][1].scatter(x_testing[:, 1], y_testing[:, 0],
                          label='Testing data')
        ax1[0][1].scatter(x_training[:, 1], y_training[:, 0],
                          label='Training data')
        fitting_w_101, fitting_b_101 = linear_fitting(x_testing[:, 1],
                                                      y_testing[:, 0])
        ax1[0][1].plot(x_Al_2, fitting_w_101 * x_Al_2 + fitting_b_101, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_101, fitting_w_101))
        ax1[0][1].legend()
        # Si/UTS
        ax1[1][0].set_xlabel('Si / wt.%')
        ax1[1][0].set_ylabel('UTS / MPa')
        ax1[1][0].set_title('Si / UTS', fontstyle='oblique')
        ax1[1][0].scatter(x_testing[:, 2], y_testing[:, 0],
                          label='Testing data')
        ax1[1][0].scatter(x_training[:, 2], y_training[:, 0],
                          label='Training data')
        fitting_w_110, fitting_b_110 = linear_fitting(x_testing[:, 2],
                                                      y_testing[:, 0])
        ax1[1][0].plot(x_Si, fitting_w_110 * x_Si + fitting_b_110, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_110, fitting_w_110))
        ax1[1][0].legend()
        # AlSi2Sr/UTS
        ax1[1][1].set_xlabel('AlSi2Sr / wt.%')
        ax1[1][1].set_ylabel('UTS / MPa')
        ax1[1][1].set_title('AlSi2Sr / UTS', fontstyle='oblique')
        ax1[1][1].scatter(x_testing[:, 3], y_testing[:, 0],
                          label='Testing data')
        ax1[1][1].scatter(x_training[:, 3], y_training[:, 0],
                          label='Training data')
        fitting_w_111, fitting_b_111 = linear_fitting(x_testing[:, 3],
                                                      y_testing[:, 0])
        ax1[1][1].plot(x_AlSi2Sr, fitting_w_111 * x_AlSi2Sr + fitting_b_111, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_111, fitting_w_111))
        ax1[1][1].legend()
        plt.savefig(path + 'phase_UTS.png', bbox_inches='tight')
        # YS
        fig2, ax2 = plt.subplots(2, 2, figsize=(16, 12))
        # Al_1/YS
        ax2[0][0].set_xlabel('Al_1 / wt.%')
        ax2[0][0].set_ylabel('YS / MPa')
        ax2[0][0].set_title('Al_1 / YS', fontstyle='oblique')
        ax2[0][0].scatter(x_testing[:, 0], y_testing[:, 1],
                          label='Testing data')
        ax2[0][0].scatter(x_training[:, 0], y_training[:, 1],
                          label='Training data')
        fitting_w_200, fitting_b_200 = linear_fitting(x_testing[:, 0],
                                                      y_testing[:, 1])
        ax2[0][0].plot(x_Al_1, fitting_w_200 * x_Al_1 + fitting_b_200, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_200, fitting_w_200))
        ax2[0][0].legend(loc='upper left')
        # Al_2/YS
        ax2[0][1].set_xlabel('Al_2 / wt.%')
        ax2[0][1].set_ylabel('YS / MPa')
        ax2[0][1].set_title('Al_2 / YS', fontstyle='oblique')
        ax2[0][1].scatter(x_testing[:, 1], y_testing[:, 1],
                          label='Testing data')
        ax2[0][1].scatter(x_training[:, 1], y_training[:, 1],
                          label='Training data')
        fitting_w_201, fitting_b_201 = linear_fitting(x_testing[:, 1],
                                                      y_testing[:, 1])
        ax2[0][1].plot(x_Al_2, fitting_w_201 * x_Al_2 + fitting_b_201, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_201, fitting_w_201))
        ax2[0][1].legend(loc='upper left')
        # Si/YS
        ax2[1][0].set_xlabel('Si / wt.%')
        ax2[1][0].set_ylabel('YS / MPa')
        ax2[1][0].set_title('Si / YS', fontstyle='oblique')
        ax2[1][0].scatter(x_testing[:, 2], y_testing[:, 1],
                          label='Testing data')
        ax2[1][0].scatter(x_training[:, 2], y_training[:, 1],
                          label='Training data')
        fitting_w_210, fitting_b_210 = linear_fitting(x_testing[:, 2],
                                                      y_testing[:, 1])
        ax2[1][0].plot(x_Si, fitting_w_210 * x_Si + fitting_b_210, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_210, fitting_w_210))
        ax2[1][0].legend(loc='upper left')
        # AlSi2Sr/YS
        ax2[1][1].set_xlabel('AlSi2Sr / wt.%')
        ax2[1][1].set_ylabel('YS / MPa')
        ax2[1][1].set_title('AlSi2Sr / YS', fontstyle='oblique')
        ax2[1][1].scatter(x_testing[:, 3], y_testing[:, 1],
                          label='Testing data')
        ax2[1][1].scatter(x_training[:, 3], y_training[:, 1],
                          label='Training data')
        fitting_w_211, fitting_b_211 = linear_fitting(x_testing[:, 3],
                                                      y_testing[:, 1])
        ax2[1][1].plot(x_AlSi2Sr, fitting_w_211 * x_AlSi2Sr + fitting_b_211, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_211, fitting_w_211))
        ax2[1][1].legend(loc='upper left')
        plt.savefig(path + 'phase_YS.png', bbox_inches='tight')
        # EL
        fig3, ax3 = plt.subplots(2, 2, figsize=(16, 12))
        # Al_1/EL
        ax3[0][0].set_xlabel('Al_1 / wt.%')
        ax3[0][0].set_ylabel('EL / %')
        ax3[0][0].set_title('Al_1 / EL', fontstyle='oblique')
        ax3[0][0].scatter(x_testing[:, 0], y_testing[:, 2],
                          label='Testing data')
        ax3[0][0].scatter(x_training[:, 0], y_training[:, 2],
                          label='Training data')
        fitting_w_300, fitting_b_300 = linear_fitting(x_testing[:, 0],
                                                      y_testing[:, 2])
        ax3[0][0].plot(x_Al_1, fitting_w_300 * x_Al_1 + fitting_b_300, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_300, fitting_w_300))
        ax3[0][0].legend()
        # Al_2/EL
        ax3[0][1].set_xlabel('Al_2 / wt.%')
        ax3[0][1].set_ylabel('EL / %')
        ax3[0][1].set_title('Al_2 / EL', fontstyle='oblique')
        ax3[0][1].scatter(x_testing[:, 1], y_testing[:, 2],
                          label='Testing data')
        ax3[0][1].scatter(x_training[:, 1], y_training[:, 2],
                          label='Training data')
        fitting_w_301, fitting_b_301 = linear_fitting(
            x_testing[:, 1], y_testing[:, 2])
        ax3[0][1].plot(x_Al_2, fitting_w_301 * x_Al_2 + fitting_b_301, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_301, fitting_w_301))
        ax3[0][1].legend()
        # Si/EL
        ax3[1][0].set_xlabel('Si / wt.%')
        ax3[1][0].set_ylabel('EL / %')
        ax3[1][0].set_title('Si / EL', fontstyle='oblique')
        ax3[1][0].scatter(x_testing[:, 2], y_testing[:, 2],
                          label='Testing data')
        ax3[1][0].scatter(x_training[:, 2], y_training[:, 2],
                          label='Training data')
        fitting_w_310, fitting_b_310 = linear_fitting(x_testing[:, 2],
                                                      y_testing[:, 2])
        ax3[1][0].plot(x_Si, fitting_w_310 * x_Si + fitting_b_310, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_310, fitting_w_310))
        ax3[1][0].legend()
        # AlSi2Sr/EL
        ax3[1][1].set_xlabel('AlSi2Sr / wt.%')
        ax3[1][1].set_ylabel('EL / %')
        ax3[1][1].set_title('AlSi2Sr / EL', fontstyle='oblique')
        ax3[1][1].scatter(x_testing[:, 3], y_testing[:, 2],
                          label='Testing data')
        ax3[1][1].scatter(x_training[:, 3], y_training[:, 2],
                          label='Training data')
        fitting_w_311, fitting_b_311 = linear_fitting(x_testing[:, 3],
                                                      y_testing[:, 2])
        ax3[1][1].plot(x_AlSi2Sr, fitting_w_311 * x_AlSi2Sr + fitting_b_311, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_311, fitting_w_311))
        ax3[1][1].legend()
        plt.savefig(path + 'phase_EL.png', bbox_inches='tight')
        linear_coef = pd.DataFrame(data=np.ones((12, 2)),
                                   index=['UTS_Al_1', 'UTS_Al_2', 'UTS_Si', 'UTS_AlSi2Sr',
                                          'YS_Al_1', 'YS_Al_2', 'YS_Si', 'YS_AlSi2Sr',
                                          'EL_Al_1', 'EL_Al_2', 'EL_Si', 'EL_AlSi2Sr'],
                                   columns=['weight', 'bias'])
        # UTS
        linear_coef.iloc[0, 0] = fitting_w_100
        linear_coef.iloc[0, 1] = fitting_b_100
        linear_coef.iloc[1, 0] = fitting_w_101
        linear_coef.iloc[1, 1] = fitting_b_101
        linear_coef.iloc[2, 0] = fitting_w_110
        linear_coef.iloc[2, 1] = fitting_b_110
        linear_coef.iloc[3, 0] = fitting_w_111
        linear_coef.iloc[3, 1] = fitting_b_111
        # YS
        linear_coef.iloc[4, 0] = fitting_w_200
        linear_coef.iloc[4, 1] = fitting_b_200
        linear_coef.iloc[5, 0] = fitting_w_201
        linear_coef.iloc[5, 1] = fitting_b_201
        linear_coef.iloc[6, 0] = fitting_w_210
        linear_coef.iloc[6, 1] = fitting_b_210
        linear_coef.iloc[7, 0] = fitting_w_211
        linear_coef.iloc[7, 1] = fitting_b_211
        # EL
        linear_coef.iloc[8, 0] = fitting_w_300
        linear_coef.iloc[8, 1] = fitting_b_300
        linear_coef.iloc[9, 0] = fitting_w_301
        linear_coef.iloc[9, 1] = fitting_b_301
        linear_coef.iloc[10, 0] = fitting_w_310
        linear_coef.iloc[10, 1] = fitting_b_310
        linear_coef.iloc[11, 0] = fitting_w_311
        linear_coef.iloc[11, 1] = fitting_b_311
        linear_coef.to_csv(path + 'linear_coef.csv', float_format='%.2f')
        # plt.show()

    # 绘制相分数-性能(归一化)关系图

    def draw_relation_performanceRE(x_training_, y_training_, x_testing_, y_testing_):
        sns.set(font="Times New Roman", font_scale=1)
        x_Al_1 = np.linspace(0.305, 0.508, 100)
        x_Al_2 = np.linspace(0, 0.401, 100)
        x_Si = np.linspace(0.046, 0.053, 100)
        x_AlSi2Sr = np.linspace(0, 0.0023, 100)
        # UTS
        fig1, ax1 = plt.subplots(2, 2, figsize=(16, 12))
        # Al_1/UTS
        ax1[0][0].set_xlabel('Al_1 / wt.%')
        ax1[0][0].set_ylabel('UTS with regularization')
        ax1[0][0].set_title('Al_1 / UTS', fontstyle='oblique')
        ax1[0][0].scatter(x_testing_[:, 0], y_testing_[:, 0],
                          label='Testing data')
        ax1[0][0].scatter(x_training_[:, 0], y_training_[:, 0],
                          label='Training data')
        fitting_w_100, fitting_b_100 = linear_fitting(x_testing_[:, 0],
                                                      y_testing_[:, 0])
        ax1[0][0].plot(x_Al_1, fitting_w_100 * x_Al_1 + fitting_b_100, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_100, fitting_w_100))
        ax1[0][0].legend()
        # Al_2/UTS
        ax1[0][1].set_xlabel('Al_2 / wt.%')
        ax1[0][1].set_ylabel('UTS with regularization')
        ax1[0][1].set_title('Al_2 / UTS', fontstyle='oblique')
        ax1[0][1].scatter(x_testing_[:, 1], y_testing_[:, 0],
                          label='Testing data')
        ax1[0][1].scatter(x_training_[:, 1], y_training_[:, 0],
                          label='Training data')
        fitting_w_101, fitting_b_101 = linear_fitting(x_testing_[:, 1],
                                                      y_testing_[:, 0])
        ax1[0][1].plot(x_Al_2, fitting_w_101 * x_Al_2 + fitting_b_101, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_101, fitting_w_101))
        ax1[0][1].legend()
        # Si/UTS
        ax1[1][0].set_xlabel('Si / wt.%')
        ax1[1][0].set_ylabel('UTS with regularization')
        ax1[1][0].set_title('Si / UTS', fontstyle='oblique')
        ax1[1][0].scatter(x_testing_[:, 2], y_testing_[:, 0],
                          label='Testing data')
        ax1[1][0].scatter(x_training_[:, 2], y_training_[:, 0],
                          label='Training data')
        fitting_w_110, fitting_b_110 = linear_fitting(x_testing_[:, 2],
                                                      y_testing_[:, 0])
        ax1[1][0].plot(x_Si, fitting_w_110 * x_Si + fitting_b_110, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_110, fitting_w_110))
        ax1[1][0].legend()
        # AlSi2Sr/UTS
        ax1[1][1].set_xlabel('AlSi2Sr / wt.%')
        ax1[1][1].set_ylabel('UTS with regularization')
        ax1[1][1].set_title('AlSi2Sr / UTS', fontstyle='oblique')
        ax1[1][1].scatter(x_testing_[:, 3], y_testing_[:, 0],
                          label='Testing data')
        ax1[1][1].scatter(x_training_[:, 3], y_training_[:, 0],
                          label='Training data')
        fitting_w_111, fitting_b_111 = linear_fitting(x_testing_[:, 3],
                                                      y_testing_[:, 0])
        ax1[1][1].plot(x_AlSi2Sr, fitting_w_111 * x_AlSi2Sr + fitting_b_111, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_111, fitting_w_111))
        ax1[1][1].legend()
        plt.savefig(path + 'phase_UTS_performanceRE.png', bbox_inches='tight')
        # YS
        fig2, ax2 = plt.subplots(2, 2, figsize=(16, 12))
        # Al_1/YS
        ax2[0][0].set_xlabel('Al_1 / wt.%')
        ax2[0][0].set_ylabel('YS with regularization')
        ax2[0][0].set_title('Al_1 / YS', fontstyle='oblique')
        ax2[0][0].scatter(x_testing_[:, 0], y_testing_[:, 1],
                          label='Testing data')
        ax2[0][0].scatter(x_training_[:, 0], y_training_[:, 1],
                          label='Training data')
        fitting_w_200, fitting_b_200 = linear_fitting(x_testing_[:, 0],
                                                      y_testing_[:, 1])
        ax2[0][0].plot(x_Al_1, fitting_w_200 * x_Al_1 + fitting_b_200, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_200, fitting_w_200))
        ax2[0][0].legend(loc='upper left')
        # Al_2/YS
        ax2[0][1].set_xlabel('Al_2 / wt.%')
        ax2[0][1].set_ylabel('YS with regularization')
        ax2[0][1].set_title('Al_2 / YS', fontstyle='oblique')
        ax2[0][1].scatter(x_testing_[:, 1], y_testing_[:, 1],
                          label='Testing data')
        ax2[0][1].scatter(x_training_[:, 1], y_training_[:, 1],
                          label='Training data')
        fitting_w_201, fitting_b_201 = linear_fitting(x_testing_[:, 1],
                                                      y_testing_[:, 1])
        ax2[0][1].plot(x_Al_2, fitting_w_201 * x_Al_2 + fitting_b_201, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_201, fitting_w_201))
        ax2[0][1].legend(loc='upper left')
        # Si/YS
        ax2[1][0].set_xlabel('Si / wt.%')
        ax2[1][0].set_ylabel('YS with regularization')
        ax2[1][0].set_title('Si / YS', fontstyle='oblique')
        ax2[1][0].scatter(x_testing_[:, 2], y_testing_[:, 1],
                          label='Testing data')
        ax2[1][0].scatter(x_training_[:, 2], y_training_[:, 1],
                          label='Training data')
        fitting_w_210, fitting_b_210 = linear_fitting(x_testing_[:, 2],
                                                      y_testing_[:, 1])
        ax2[1][0].plot(x_Si, fitting_w_210 * x_Si + fitting_b_210, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_210, fitting_w_210))
        ax2[1][0].legend(loc='upper left')
        # AlSi2Sr/YS
        ax2[1][1].set_xlabel('AlSi2Sr / wt.%')
        ax2[1][1].set_ylabel('YS with regularization')
        ax2[1][1].set_title('AlSi2Sr / YS', fontstyle='oblique')
        ax2[1][1].scatter(x_testing_[:, 3], y_testing_[:, 1],
                          label='Testing data')
        ax2[1][1].scatter(x_training_[:, 3], y_training_[:, 1],
                          label='Training data')
        fitting_w_211, fitting_b_211 = linear_fitting(x_testing_[:, 3],
                                                      y_testing_[:, 1])
        ax2[1][1].plot(x_AlSi2Sr, fitting_w_211 * x_AlSi2Sr + fitting_b_211, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_211, fitting_w_211))
        ax2[1][1].legend(loc='upper left')
        plt.savefig(path + 'phase_YS_performanceRE.png', bbox_inches='tight')
        # EL
        fig3, ax3 = plt.subplots(2, 2, figsize=(16, 12))
        # Al_1/EL
        ax3[0][0].set_xlabel('Al_1 / wt.%')
        ax3[0][0].set_ylabel('EL with regularization')
        ax3[0][0].set_title('Al_1 / EL', fontstyle='oblique')
        ax3[0][0].scatter(x_testing_[:, 0], y_testing_[:, 2],
                          label='Testing data')
        ax3[0][0].scatter(x_training_[:, 0], y_training_[:, 2],
                          label='Training data')
        fitting_w_300, fitting_b_300 = linear_fitting(x_testing_[:, 0],
                                                      y_testing_[:, 2])
        ax3[0][0].plot(x_Al_1, fitting_w_300 * x_Al_1 + fitting_b_300, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_300, fitting_w_300))
        ax3[0][0].legend()
        # Al_2/EL
        ax3[0][1].set_xlabel('Al_2 / wt.%')
        ax3[0][1].set_ylabel('EL with regularization')
        ax3[0][1].set_title('Al_2 / EL', fontstyle='oblique')
        ax3[0][1].scatter(x_testing_[:, 1], y_testing_[:, 2],
                          label='Testing data')
        ax3[0][1].scatter(x_training_[:, 1], y_training_[:, 2],
                          label='Training data')
        fitting_w_301, fitting_b_301 = linear_fitting(x_testing_[:, 1],
                                                      y_testing_[:, 2])
        ax3[0][1].plot(x_Al_2, fitting_w_301 * x_Al_2 + fitting_b_301, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_301, fitting_w_301))
        ax3[0][1].legend()
        # Si/EL
        ax3[1][0].set_xlabel('Si / wt.%')
        ax3[1][0].set_ylabel('EL with regularization')
        ax3[1][0].set_title('Si / EL', fontstyle='oblique')
        ax3[1][0].scatter(x_testing_[:, 2], y_testing_[:, 2],
                          label='Testing data')
        ax3[1][0].scatter(x_training_[:, 2], y_training_[:, 2],
                          label='Training data')
        fitting_w_310, fitting_b_310 = linear_fitting(x_testing_[:, 2],
                                                      y_testing_[:, 2])
        ax3[1][0].plot(x_Si, fitting_w_310 * x_Si + fitting_b_310, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_310, fitting_w_310))
        ax3[1][0].legend()
        # AlSi2Sr/EL
        ax3[1][1].set_xlabel('AlSi2Sr / wt.%')
        ax3[1][1].set_ylabel('EL with regularization')
        ax3[1][1].set_title('AlSi2Sr / EL', fontstyle='oblique')
        ax3[1][1].scatter(x_testing_[:, 3], y_testing_[:, 2],
                          label='Testing data')
        ax3[1][1].scatter(x_training_[:, 3], y_training_[:, 2],
                          label='Training data')
        fitting_w_311, fitting_b_311 = linear_fitting(x_testing_[:, 3],
                                                      y_testing_[:, 2])
        ax3[1][1].plot(x_AlSi2Sr, fitting_w_311 * x_AlSi2Sr + fitting_b_311, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_311, fitting_w_311))
        ax3[1][1].legend()
        plt.savefig(path + 'phase_EL_performanceRE.png', bbox_inches='tight')
        linear_coef_allRE = pd.DataFrame(data=np.ones((12, 2)),
                                         index=['UTS_Al_1', 'UTS_Al_2', 'UTS_Si', 'UTS_AlSi2Sr',
                                                'YS_Al_1', 'YS_Al_2', 'YS_Si', 'YS_AlSi2Sr',
                                                'EL_Al_1', 'EL_Al_2', 'EL_Si', 'EL_AlSi2Sr'],
                                         columns=['weight', 'bias'])
        # UTS
        linear_coef_allRE.iloc[0, 0] = fitting_w_100
        linear_coef_allRE.iloc[0, 1] = fitting_b_100
        linear_coef_allRE.iloc[1, 0] = fitting_w_101
        linear_coef_allRE.iloc[1, 1] = fitting_b_101
        linear_coef_allRE.iloc[2, 0] = fitting_w_110
        linear_coef_allRE.iloc[2, 1] = fitting_b_110
        linear_coef_allRE.iloc[3, 0] = fitting_w_111
        linear_coef_allRE.iloc[3, 1] = fitting_b_111
        # YS
        linear_coef_allRE.iloc[4, 0] = fitting_w_200
        linear_coef_allRE.iloc[4, 1] = fitting_b_200
        linear_coef_allRE.iloc[5, 0] = fitting_w_201
        linear_coef_allRE.iloc[5, 1] = fitting_b_201
        linear_coef_allRE.iloc[6, 0] = fitting_w_210
        linear_coef_allRE.iloc[6, 1] = fitting_b_210
        linear_coef_allRE.iloc[7, 0] = fitting_w_211
        linear_coef_allRE.iloc[7, 1] = fitting_b_211
        # EL
        linear_coef_allRE.iloc[8, 0] = fitting_w_300
        linear_coef_allRE.iloc[8, 1] = fitting_b_300
        linear_coef_allRE.iloc[9, 0] = fitting_w_301
        linear_coef_allRE.iloc[9, 1] = fitting_b_301
        linear_coef_allRE.iloc[10, 0] = fitting_w_310
        linear_coef_allRE.iloc[10, 1] = fitting_b_310
        linear_coef_allRE.iloc[11, 0] = fitting_w_311
        linear_coef_allRE.iloc[11, 1] = fitting_b_311
        linear_coef_allRE.to_csv(
            path + 'linear_coef_performanceRE.csv', float_format='%.2f')
        # plt.show()

    # 绘制相分数(归一化)-性能(归一化)关系图

    def draw_relation_allRE(x_training_, y_training_, x_testing_, y_testing_):
        sns.set(font="Times New Roman", font_scale=1)
        x_Al_1 = np.linspace(-2.3, 2.3, 100)
        x_Al_2 = np.linspace(-2.3, 2.3, 100)
        x_Si = np.linspace(-2.3, 2.3, 100)
        x_AlSi2Sr = np.linspace(-2.3, 2.3, 100)
        # UTS
        fig1, ax1 = plt.subplots(2, 2, figsize=(16, 12))
        # Al_1/UTS
        ax1[0][0].set_xlabel('Al_1 with regularization')
        ax1[0][0].set_ylabel('UTS with regularization')
        ax1[0][0].set_title('Al_1 / UTS', fontstyle='oblique')
        ax1[0][0].scatter(x_testing_[:, 0], y_testing_[:, 0],
                          label='Testing data')
        ax1[0][0].scatter(x_training_[:, 0], y_training_[:, 0],
                          label='Training data')
        fitting_w_100, fitting_b_100 = linear_fitting(x_testing_[:, 0],
                                                      y_testing_[:, 0])
        ax1[0][0].plot(x_Al_1, fitting_w_100 * x_Al_1 + fitting_b_100, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_100, fitting_w_100))
        ax1[0][0].legend()
        # Al_2/UTS
        ax1[0][1].set_xlabel('Al_2 with regularization')
        ax1[0][1].set_ylabel('UTS with regularization')
        ax1[0][1].set_title('Al_2 / UTS', fontstyle='oblique')
        ax1[0][1].scatter(x_testing_[:, 1], y_testing_[:, 0],
                          label='Testing data')
        ax1[0][1].scatter(x_training_[:, 1], y_training_[:, 0],
                          label='Training data')
        fitting_w_101, fitting_b_101 = linear_fitting(x_testing_[:, 1],
                                                      y_testing_[:, 0])
        ax1[0][1].plot(x_Al_2, fitting_w_101 * x_Al_2 + fitting_b_101, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_101, fitting_w_101))
        ax1[0][1].legend()
        # Si/UTS
        ax1[1][0].set_xlabel('Si with regularization')
        ax1[1][0].set_ylabel('UTS with regularization')
        ax1[1][0].set_title('Si / UTS', fontstyle='oblique')
        ax1[1][0].scatter(x_testing_[:, 2], y_testing_[:, 0],
                          label='Testing data')
        ax1[1][0].scatter(x_training_[:, 2], y_training_[:, 0],
                          label='Training data')
        fitting_w_110, fitting_b_110 = linear_fitting(x_testing_[:, 2],
                                                      y_testing_[:, 0])
        ax1[1][0].plot(x_Si, fitting_w_110 * x_Si + fitting_b_110, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_110, fitting_w_110))
        ax1[1][0].legend()
        # AlSi2Sr/UTS
        ax1[1][1].set_xlabel('AlSi2Sr with regularization')
        ax1[1][1].set_ylabel('UTS with regularization')
        ax1[1][1].set_title('AlSi2Sr / UTS', fontstyle='oblique')
        ax1[1][1].scatter(x_testing_[:, 2], y_testing_[:, 0],
                          label='Testing data')
        ax1[1][1].scatter(x_training_[:, 2], y_training_[:, 0],
                          label='Training data')
        fitting_w_111, fitting_b_111 = linear_fitting(x_testing_[:, 2],
                                                      y_testing_[:, 0])
        ax1[1][1].plot(x_AlSi2Sr, fitting_w_111 * x_AlSi2Sr + fitting_b_111, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_111, fitting_w_111))
        ax1[1][1].legend()
        # AlSi2Sr/UTS
        ax1[1][1].set_xlabel('AlSi2Sr with regularization')
        ax1[1][1].set_ylabel('UTS with regularization')
        ax1[1][1].set_title('AlSi2Sr / UTS', fontstyle='oblique')
        ax1[1][1].scatter(x_testing_[:, 3], y_testing_[:, 0],
                          label='Testing data')
        ax1[1][1].scatter(x_training_[:, 3], y_training_[:, 0],
                          label='Training data')
        fitting_w_111, fitting_b_111 = linear_fitting(x_testing_[:, 3],
                                                      y_testing_[:, 0])
        ax1[1][1].plot(x_AlSi2Sr, fitting_w_111 * x_AlSi2Sr + fitting_b_111, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_111, fitting_w_111))
        ax1[1][1].legend()
        plt.savefig(path + 'phase_UTS_allRE.png', bbox_inches='tight')
        # YS
        fig2, ax2 = plt.subplots(2, 2, figsize=(16, 12))
        # Al_1/YS
        ax2[0][0].set_xlabel('Al_1 with regularization')
        ax2[0][0].set_ylabel('YS with regularization')
        ax2[0][0].set_title('Al_1 / YS', fontstyle='oblique')
        ax2[0][0].scatter(x_testing_[:, 0], y_testing_[:, 1],
                          label='Testing data')
        ax2[0][0].scatter(x_training_[:, 0], y_training_[:, 1],
                          label='Training data')
        fitting_w_200, fitting_b_200 = linear_fitting(x_testing_[:, 0],
                                                      y_testing_[:, 1])
        ax2[0][0].plot(x_Al_1, fitting_w_200 * x_Al_1 + fitting_b_200, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_200, fitting_w_200))
        ax2[0][0].legend(loc='upper left')
        # Al_2/YS
        ax2[0][1].set_xlabel('Al_2 with regularization')
        ax2[0][1].set_ylabel('YS with regularization')
        ax2[0][1].set_title('Al_2 / YS', fontstyle='oblique')
        ax2[0][1].scatter(x_testing_[:, 1], y_testing_[:, 1],
                          label='Testing data')
        ax2[0][1].scatter(x_training_[:, 1], y_training_[:, 1],
                          label='Training data')
        fitting_w_201, fitting_b_201 = linear_fitting(x_testing_[:, 1],
                                                      y_testing_[:, 1])
        ax2[0][1].plot(x_Al_2, fitting_w_201 * x_Al_2 + fitting_b_201, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_201, fitting_w_201))
        ax2[0][1].legend(loc='upper left')
        # Si/YS
        ax2[1][0].set_xlabel('Si with regularization')
        ax2[1][0].set_ylabel('YS with regularization')
        ax2[1][0].set_title('Si / YS', fontstyle='oblique')
        ax2[1][0].scatter(x_testing_[:, 2], y_testing_[:, 1],
                          label='Testing data')
        ax2[1][0].scatter(x_training_[:, 2], y_training_[:, 1],
                          label='Training data')
        fitting_w_210, fitting_b_210 = linear_fitting(x_testing_[:, 2],
                                                      y_testing_[:, 1])
        ax2[1][0].plot(x_Si, fitting_w_210 * x_Si + fitting_b_210, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_210, fitting_w_210))
        ax2[1][0].legend(loc='upper left')
        # AlSi2Sr/YS
        ax2[1][1].set_xlabel('AlSi2Sr with regularization')
        ax2[1][1].set_ylabel('YS with regularization')
        ax2[1][1].set_title('AlSi2Sr / YS', fontstyle='oblique')
        ax2[1][1].scatter(x_testing_[:, 3], y_testing_[:, 1],
                          label='Testing data')
        ax2[1][1].scatter(x_training_[:, 3], y_training_[:, 1],
                          label='Training data')
        fitting_w_211, fitting_b_211 = linear_fitting(x_testing_[:, 3],
                                                      y_testing_[:, 1])
        ax2[1][1].plot(x_AlSi2Sr, fitting_w_211 * x_AlSi2Sr + fitting_b_211, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_211, fitting_w_211))
        ax2[1][1].legend(loc='upper left')
        plt.savefig(path + 'phase_YS_allRE.png', bbox_inches='tight')
        # EL
        fig3, ax3 = plt.subplots(2, 2, figsize=(16, 12))
        # Al_1/EL
        ax3[0][0].set_xlabel('Al_1 with regularization')
        ax3[0][0].set_ylabel('EL with regularization')
        ax3[0][0].set_title('Al_1 / EL', fontstyle='oblique')
        ax3[0][0].scatter(x_testing_[:, 0], y_testing_[:, 2],
                          label='Testing data')
        ax3[0][0].scatter(x_training_[:, 0], y_training_[:, 2],
                          label='Training data')
        fitting_w_300, fitting_b_300 = linear_fitting(x_testing_[:, 0],
                                                      y_testing_[:, 2])
        ax3[0][0].plot(x_Al_1, fitting_w_300 * x_Al_1 + fitting_b_300, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_300, fitting_w_300))
        ax3[0][0].legend()
        # Al_2/EL
        ax3[0][1].set_xlabel('Al_2 with regularization')
        ax3[0][1].set_ylabel('EL with regularization')
        ax3[0][1].set_title('Al_2 / EL', fontstyle='oblique')
        ax3[0][1].scatter(x_testing_[:, 1], y_testing_[:, 2],
                          label='Testing data')
        ax3[0][1].scatter(x_training_[:, 1], y_training_[:, 2],
                          label='Training data')
        fitting_w_301, fitting_b_301 = linear_fitting(x_testing_[:, 1],
                                                      y_testing_[:, 2])
        ax3[0][1].plot(x_Al_2, fitting_w_301 * x_Al_2 + fitting_b_301, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_301, fitting_w_301))
        ax3[0][1].legend()
        # Si/EL
        ax3[1][0].set_xlabel('Si with regularization')
        ax3[1][0].set_ylabel('EL with regularization')
        ax3[1][0].set_title('Si / EL', fontstyle='oblique')
        ax3[1][0].scatter(x_testing_[:, 2], y_testing_[:, 2],
                          label='Testing data')
        ax3[1][0].scatter(x_training_[:, 2], y_training_[:, 2],
                          label='Training data')
        fitting_w_310, fitting_b_310 = linear_fitting(x_testing_[:, 2],
                                                      y_testing_[:, 2])
        ax3[1][0].plot(x_Si, fitting_w_310 * x_Si + fitting_b_310, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_310, fitting_w_310))
        ax3[1][0].legend()
        # AlSi2Sr/EL
        ax3[1][1].set_xlabel('AlSi2Sr with regularization')
        ax3[1][1].set_ylabel('EL with regularization')
        ax3[1][1].set_title('AlSi2Sr / EL', fontstyle='oblique')
        ax3[1][1].scatter(x_testing_[:, 3], y_testing_[:, 2],
                          label='Testing data')
        ax3[1][1].scatter(x_training_[:, 3], y_training_[:, 2],
                          label='Training data')
        fitting_w_311, fitting_b_311 = linear_fitting(x_testing_[:, 3],
                                                      y_testing_[:, 2])
        ax3[1][1].plot(x_AlSi2Sr, fitting_w_311 * x_AlSi2Sr + fitting_b_311, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_311, fitting_w_311))
        ax3[1][1].legend()
        plt.savefig(path + 'phase_EL_allRE.png', bbox_inches='tight')
        linear_coef_allRE = pd.DataFrame(data=np.ones((12, 2)),
                                         index=['UTS_Al_1', 'UTS_Al_2', 'UTS_Si', 'UTS_AlSi2Sr',
                                                'YS_Al_1', 'YS_Al_2', 'YS_Si', 'YS_AlSi2Sr',
                                                'EL_Al_1', 'EL_Al_2', 'EL_Si', 'EL_AlSi2Sr'],
                                         columns=['weight', 'bias'])
        # UTS
        linear_coef_allRE.iloc[0, 0] = fitting_w_100
        linear_coef_allRE.iloc[0, 1] = fitting_b_100
        linear_coef_allRE.iloc[1, 0] = fitting_w_101
        linear_coef_allRE.iloc[1, 1] = fitting_b_101
        linear_coef_allRE.iloc[2, 0] = fitting_w_110
        linear_coef_allRE.iloc[2, 1] = fitting_b_110
        linear_coef_allRE.iloc[3, 0] = fitting_w_111
        linear_coef_allRE.iloc[3, 1] = fitting_b_111
        # YS
        linear_coef_allRE.iloc[4, 0] = fitting_w_200
        linear_coef_allRE.iloc[4, 1] = fitting_b_200
        linear_coef_allRE.iloc[5, 0] = fitting_w_201
        linear_coef_allRE.iloc[5, 1] = fitting_b_201
        linear_coef_allRE.iloc[6, 0] = fitting_w_210
        linear_coef_allRE.iloc[6, 1] = fitting_b_210
        linear_coef_allRE.iloc[7, 0] = fitting_w_211
        linear_coef_allRE.iloc[7, 1] = fitting_b_211
        # EL
        linear_coef_allRE.iloc[8, 0] = fitting_w_300
        linear_coef_allRE.iloc[8, 1] = fitting_b_300
        linear_coef_allRE.iloc[9, 0] = fitting_w_301
        linear_coef_allRE.iloc[9, 1] = fitting_b_301
        linear_coef_allRE.iloc[10, 0] = fitting_w_310
        linear_coef_allRE.iloc[10, 1] = fitting_b_310
        linear_coef_allRE.iloc[11, 0] = fitting_w_311
        linear_coef_allRE.iloc[11, 1] = fitting_b_311
        linear_coef_allRE.to_csv(
            path + 'linear_coef_allRE.csv', float_format='%.2f')
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
        if np.isnan(y_predicting.numpy().any()):
            print('==============Prediction run out of range===============')
        else:
            print('==================Prediction complete===================')

            # 数据可视化(散点图)
            draw_scatter(EL_Sr.numpy(), y_UTS.numpy(),
                         EL_Sr_predicting.numpy(), y_predicting.numpy()[:, 0],
                         'UTS / MPa', training_accuracy[0], testing_accuracy[0])
            draw_scatter(EL_Sr.numpy(), y_YS.numpy(),
                         EL_Sr_predicting.numpy(), y_predicting.numpy()[:, 1],
                         'YS / MPa', training_accuracy[1], testing_accuracy[1])
            draw_scatter(EL_Sr.numpy(), y_EL.numpy(),
                         EL_Sr_predicting.numpy(), y_predicting.numpy()[:, 2],
                         'EL / %', training_accuracy[2], testing_accuracy[2])

            # 综合力学性能计算及可视化
            # data_process(path, EL_Si_predicting.numpy(), EL_Mg_predicting.numpy())

            # # 绘制相分数-性能关系图
            draw_relation(x.numpy(), y_list.numpy(),
                          x_predicting.numpy(), y_predicting.numpy())

            # # 绘制相分数-性能(正则化)关系图
            draw_relation_performanceRE(x.numpy(), y_standarded.numpy(),
                                        x_predicting.numpy(), y_standarded_predict.numpy())

            # 绘制相分数(正则化)-性能(正则化)关系图
            draw_relation_allRE(x_standarded.numpy(), y_standarded.numpy(),
                                x_standarded_predict.numpy(), y_standarded_predict.numpy())


if __name__ == '__main__':
    main(parameters_list)
