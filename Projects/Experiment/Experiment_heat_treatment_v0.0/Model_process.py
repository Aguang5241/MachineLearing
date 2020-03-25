###########################################################
#                net shape: n-10-5-3                      #
#                layer function: Linear                   #
#                standard: True                           #
#                activation function: ReLU                #
#                loss function: MSELoss                   #
#                optimizer: Adam                          #
###########################################################

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
    training_data_file_path = parameters_list[1]
    testing_data_file_path = parameters_list[2]
    model_path = parameters_list[3]

    # 定义获取训练数据函数

    def get_training_data(file_path):
        data = pd.read_csv(file_path)
        x = torch.from_numpy(data.loc[:, 'PH_Al':'PH_AlSc2Si2'].values).float()
        y_UTS = torch.unsqueeze(
            (torch.from_numpy(data['UTS'].values)), dim=1).float()
        y_YS = torch.unsqueeze(
            (torch.from_numpy(data['YS'].values)), dim=1).float()
        y_EL = torch.unsqueeze(
            (torch.from_numpy(data['EL'].values)), dim=1).float()
        EL_Si = torch.unsqueeze(
            (torch.from_numpy(data['EL_Si'].values)), dim=1).float()
        EL_Mg = torch.unsqueeze(
            (torch.from_numpy(data['EL_Mg'].values)), dim=1).float()
        return x, y_UTS, y_YS, y_EL, EL_Si, EL_Mg

    # 定义获取测试数据函数

    def get_testing_data(file_path):
        data = pd.read_csv(file_path)
        x = torch.from_numpy(data.loc[:, 'PH_Al':'PH_AlSc2Si2'].values).float()
        EL_Si = torch.unsqueeze(
            (torch.from_numpy(data['EL_Si'].values)), dim=1).float()
        EL_Mg = torch.unsqueeze(
            (torch.from_numpy(data['EL_Mg'].values)), dim=1).float()
        return x, EL_Si, EL_Mg

    # 定义预测函数

    def test(model_path, x):
        net = torch.load(model_path)
        predict_y = net(x)
        return predict_y

    # 综合处理全部数据

    def data_process(path, x, y):
        data = pd.read_csv(path + 'testing_results.csv')
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
        ax.scatter(x_training, y_training, z_training,
                   color='red', s=50, label='Training data')
        ax.scatter(x_testing, y_testing, z_testing, label='Testing data')
        ax.legend(loc='upper left')
        plt.savefig(path + 'elements_%s.png' % fig_name)
        # plt.show()

    # 线性拟合函数

    def linear_fitting(x, y):
        parameters = np.polyfit(x, y, 1)
        return parameters[0], parameters[1]

    # 绘制相分数-性能关系图

    def draw_relation(x_training, y_training, x_testing, y_testing):
        sns.set(font="Times New Roman", font_scale=1)
        x_Al_1 = np.linspace(0, 0.8, 100)
        x_Al_2_Si = np.linspace(0.1, 0.9, 100)
        x_AlSc2Si2 = np.linspace(0, 0.02, 100)
        # UTS
        fig1, ax1 = plt.subplots(1, 3, figsize=(18, 4.5))
        # Al_1/UTS
        ax1[0].set_xlabel('Al_1 / wt.%')
        ax1[0].set_ylabel('UTS / MPa')
        ax1[0].set_title('Al_1 / UTS', fontstyle='oblique')
        ax1[0].scatter(x_testing[:, 0], y_testing[:, 0],
                       label='Testing data')
        ax1[0].scatter(x_training[:, 0], y_training[:, 0],
                       label='Training data')
        fitting_w_10, fitting_b_10 = linear_fitting(x_testing[:, 0],
                                                    y_testing[:, 0])
        ax1[0].plot(x_Al_1, fitting_w_10 * x_Al_1 + fitting_b_10, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_10, fitting_w_10))
        ax1[0].legend()
        # Al_2/UTS
        ax1[1].set_xlabel('Al_2 + Si / wt.%')
        ax1[1].set_ylabel('UTS / MPa')
        ax1[1].set_title('Al_2 + Si / UTS', fontstyle='oblique')
        ax1[1].scatter(x_testing[:, 1], y_testing[:, 0],
                       label='Testing data')
        ax1[1].scatter(x_training[:, 1], y_training[:, 0],
                       label='Training data')
        fitting_w_11, fitting_b_11 = linear_fitting(x_testing[:, 1],
                                                    y_testing[:, 0])
        ax1[1].plot(x_Al_2_Si, fitting_w_11 * x_Al_2_Si + fitting_b_11, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_11, fitting_w_11))
        ax1[1].legend()
        # AlSc2Si2/UTS
        ax1[2].set_xlabel('AlSc2Si2 / wt.%')
        ax1[2].set_ylabel('UTS / MPa')
        ax1[2].set_title('AlSc2Si2 / UTS', fontstyle='oblique')
        ax1[2].scatter(x_testing[:, 2], y_testing[:, 0],
                       label='Testing data')
        ax1[2].scatter(x_training[:, 2], y_training[:, 0],
                       label='Training data')
        fitting_w_12, fitting_b_12 = linear_fitting(x_testing[:, 2],
                                                    y_testing[:, 0])
        ax1[2].plot(x_AlSc2Si2, fitting_w_12 * x_AlSc2Si2 + fitting_b_12, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_12, fitting_w_12))
        ax1[2].legend()
        plt.savefig(path + 'phase_UTS.png', bbox_inches='tight')
        # YS
        fig2, ax2 = plt.subplots(1, 3, figsize=(18, 4.5))
        # Al_1/YS
        ax2[0].set_xlabel('Al_1 / wt.%')
        ax2[0].set_ylabel('YS / MPa')
        ax2[0].set_title('Al_1 / YS', fontstyle='oblique')
        ax2[0].scatter(x_testing[:, 0], y_testing[:, 1],
                       label='Testing data')
        ax2[0].scatter(x_training[:, 0], y_training[:, 1],
                       label='Training data')
        fitting_w_20, fitting_b_20 = linear_fitting(x_testing[:, 0],
                                                    y_testing[:, 1])
        ax2[0].plot(x_Al_1, fitting_w_20 * x_Al_1 + fitting_b_20, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_20, fitting_w_20))
        ax2[0].legend(loc='upper left')
        # Al_2/YS
        ax2[1].set_xlabel('Al_2 + Si / wt.%')
        ax2[1].set_ylabel('YS / MPa')
        ax2[1].set_title('Al_2 + Si / YS', fontstyle='oblique')
        ax2[1].scatter(x_testing[:, 1], y_testing[:, 1],
                       label='Testing data')
        ax2[1].scatter(x_training[:, 1], y_training[:, 1],
                       label='Training data')
        fitting_w_21, fitting_b_21 = linear_fitting(x_testing[:, 1],
                                                    y_testing[:, 1])
        ax2[1].plot(x_Al_2_Si, fitting_w_21 * x_Al_2_Si + fitting_b_21, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_21, fitting_w_21))
        ax2[1].legend(loc='upper left')
        # AlSc2Si2/YS
        ax2[2].set_xlabel('AlSc2Si2 / wt.%')
        ax2[2].set_ylabel('YS / MPa')
        ax2[2].set_title('AlSc2Si2 / YS', fontstyle='oblique')
        ax2[2].scatter(x_testing[:, 2], y_testing[:, 1],
                       label='Testing data')
        ax2[2].scatter(x_training[:, 2], y_training[:, 1],
                       label='Training data')
        fitting_w_22, fitting_b_22 = linear_fitting(x_testing[:, 2],
                                                    y_testing[:, 1])
        ax2[2].plot(x_AlSc2Si2, fitting_w_22 * x_AlSc2Si2 + fitting_b_22, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_22, fitting_w_22))
        ax2[2].legend(loc='upper left')
        plt.savefig(path + 'phase_YS.png', bbox_inches='tight')
        # EL
        fig3, ax3 = plt.subplots(1, 3, figsize=(18, 4.5))
        # Al_1/EL
        ax3[0].set_xlabel('Al_1 / wt.%')
        ax3[0].set_ylabel('EL / %')
        ax3[0].set_title('Al_1 / EL', fontstyle='oblique')
        ax3[0].scatter(x_testing[:, 0], y_testing[:, 2],
                       label='Testing data')
        ax3[0].scatter(x_training[:, 0], y_training[:, 2],
                       label='Training data')
        fitting_w_30, fitting_b_30 = linear_fitting(x_testing[:, 0],
                                                    y_testing[:, 2])
        ax3[0].plot(x_Al_1, fitting_w_30 * x_Al_1 + fitting_b_30, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_30, fitting_w_30))
        ax3[0].legend()
        # Al_2/EL
        ax3[1].set_xlabel('Al_2 + Si / wt.%')
        ax3[1].set_ylabel('EL / %')
        ax3[1].set_title('Al_2 + Si / EL', fontstyle='oblique')
        ax3[1].scatter(x_testing[:, 1], y_testing[:, 2],
                       label='Testing data')
        ax3[1].scatter(x_training[:, 1], y_training[:, 2],
                       label='Training data')
        fitting_w_31, fitting_b_31 = linear_fitting(x_testing[:, 1],
                                                    y_testing[:, 2])
        ax3[1].plot(x_Al_2_Si, fitting_w_31 * x_Al_2_Si + fitting_b_31, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_31, fitting_w_31))
        ax3[1].legend()
        # AlSc2Si2/EL
        ax3[2].set_xlabel('AlSc2Si2 / wt.%')
        ax3[2].set_ylabel('EL / %')
        ax3[2].set_title('AlSc2Si2 / EL', fontstyle='oblique')
        ax3[2].scatter(x_testing[:, 2], y_testing[:, 2],
                       label='Testing data')
        ax3[2].scatter(x_training[:, 2], y_training[:, 2],
                       label='Training data')
        fitting_w_32, fitting_b_32 = linear_fitting(x_testing[:, 2],
                                                    y_testing[:, 2])
        ax3[2].plot(x_AlSc2Si2, fitting_w_32 * x_AlSc2Si2 + fitting_b_32, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_32, fitting_w_32))
        ax3[2].legend()
        plt.savefig(path + 'phase_EL.png', bbox_inches='tight')
        linear_coef = pd.DataFrame(data=np.ones((9, 2)),
                                   index=['UTS_Al_1', 'UTS_Al_2+Si', 'UTS_AlSc2Si2', 'YS_Al_1',
                                          'YS_Al_2+Si', 'YS_AlSc2Si2', 'EL_Al_1', 'EL_Al_2+Si', 'EL_AlSc2Si2'], columns=['weight', 'bias'])
        # UTS
        linear_coef.iloc[0, 0] = fitting_w_10
        linear_coef.iloc[0, 1] = fitting_b_10
        linear_coef.iloc[1, 0] = fitting_w_11
        linear_coef.iloc[1, 1] = fitting_b_11
        linear_coef.iloc[2, 0] = fitting_w_12
        linear_coef.iloc[2, 1] = fitting_b_12
        # YS
        linear_coef.iloc[3, 0] = fitting_w_20
        linear_coef.iloc[3, 1] = fitting_b_20
        linear_coef.iloc[4, 0] = fitting_w_21
        linear_coef.iloc[4, 1] = fitting_b_21
        linear_coef.iloc[5, 0] = fitting_w_22
        linear_coef.iloc[5, 1] = fitting_b_22
        # EL
        linear_coef.iloc[6, 0] = fitting_w_30
        linear_coef.iloc[6, 1] = fitting_b_30
        linear_coef.iloc[7, 0] = fitting_w_31
        linear_coef.iloc[7, 1] = fitting_b_31
        linear_coef.iloc[8, 0] = fitting_w_32
        linear_coef.iloc[8, 1] = fitting_b_32
        linear_coef.to_csv(path + 'linear_coef.csv', float_format='%.2f')
        # plt.show()

    # 绘制相分数(归一化)-性能关系图

    def draw_relation_phaseRE(x_training_, y_training_, x_testing_, y_testing_):
        sns.set(font="Times New Roman", font_scale=1)
        x_Al_1 = np.linspace(-2.3, 2.3, 100)
        x_Al_2 = np.linspace(-2.3, 2.3, 100)
        x_Si = np.linspace(-2.3, 2.3, 100)
        x_AlSc2Si2 = np.linspace(-2.3, 2.3, 100)
        # UTS
        fig1, ax1 = plt.subplots(2, 2, figsize=(16, 12))
        # Al_1/UTS
        ax1[0][0].set_xlabel('Al_1 with regularization')
        ax1[0][0].set_ylabel('UTS / MPa')
        ax1[0][0].set_title('Al_1 / UTS', fontstyle='oblique')
        ax1[0][0].scatter(x_testing_[:, 0], y_testing_[
                          :, 0], label='Testing data')
        ax1[0][0].scatter(x_training_[:, 0], y_training_[:, 0],
                          label='Training data')
        fitting_w_100, fitting_b_100 = linear_fitting(
            x_testing_[:, 0], y_testing_[:, 0])
        ax1[0][0].plot(x_Al_1, fitting_w_100 * x_Al_1 + fitting_b_100, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_100, fitting_w_100))
        ax1[0][0].legend()
        # Al_2/UTS
        ax1[0][1].set_xlabel('Al_2 with regularization')
        ax1[0][1].set_ylabel('UTS / MPa')
        ax1[0][1].set_title('Al_2 / UTS', fontstyle='oblique')
        ax1[0][1].scatter(x_testing_[:, 1], y_testing_[
                          :, 0], label='Testing data')
        ax1[0][1].scatter(x_training_[:, 1], y_training_[:, 0],
                          label='Training data')
        fitting_w_101, fitting_b_101 = linear_fitting(
            x_testing_[:, 1], y_testing_[:, 0])
        ax1[0][1].plot(x_Al_2, fitting_w_101 * x_Al_2 + fitting_b_101, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_101, fitting_w_101))
        ax1[0][1].legend()
        # Si/UTS
        ax1[1][0].set_xlabel('Si with regularization')
        ax1[1][0].set_ylabel('UTS / MPa')
        ax1[1][0].set_title('Si / UTS', fontstyle='oblique')
        ax1[1][0].scatter(x_testing_[:, 2], y_testing_[
                          :, 0], label='Testing data')
        ax1[1][0].scatter(x_training_[:, 2], y_training_[:, 0],
                          label='Training data')
        fitting_w_110, fitting_b_110 = linear_fitting(
            x_testing_[:, 2], y_testing_[:, 0])
        ax1[1][0].plot(x_Si, fitting_w_110 * x_Si + fitting_b_110, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_110, fitting_w_110))
        ax1[1][0].legend()
        # AlSc2Si2/UTS
        ax1[1][1].set_xlabel('AlSc2Si2 with regularization')
        ax1[1][1].set_ylabel('UTS / MPa')
        ax1[1][1].set_title('AlSc2Si2 / UTS', fontstyle='oblique')
        ax1[1][1].scatter(x_testing_[:, 3], y_testing_[
                          :, 0], label='Testing data')
        ax1[1][1].scatter(x_training_[:, 3], y_training_[:, 0],
                          label='Training data')
        fitting_w_111, fitting_b_111 = linear_fitting(
            x_testing_[:, 3], y_testing_[:, 0])
        ax1[1][1].plot(x_AlSc2Si2, fitting_w_111 * x_AlSc2Si2 + fitting_b_111, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_111, fitting_w_111))
        ax1[1][1].legend()
        plt.savefig(path + 'phase_UTS_phaseRE.png', bbox_inches='tight')
        # YS
        fig2, ax2 = plt.subplots(2, 2, figsize=(16, 12))
        # Al_1/YS
        ax2[0][0].set_xlabel('Al_1 with regularization')
        ax2[0][0].set_ylabel('YS / MPa')
        ax2[0][0].set_title('Al_1 / YS', fontstyle='oblique')
        ax2[0][0].scatter(x_testing_[:, 0], y_testing_[
                          :, 1], label='Testing data')
        ax2[0][0].scatter(x_training_[:, 0], y_training_[:, 1],
                          label='Training data')
        fitting_w_200, fitting_b_200 = linear_fitting(
            x_testing_[:, 0], y_testing_[:, 1])
        ax2[0][0].plot(x_Al_1, fitting_w_200 * x_Al_1 + fitting_b_200, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_200, fitting_w_200))
        ax2[0][0].legend(loc='upper left')
        # Al_2/YS
        ax2[0][1].set_xlabel('Al_2 with regularization')
        ax2[0][1].set_ylabel('YS / MPa')
        ax2[0][1].set_title('Al_2 / YS', fontstyle='oblique')
        ax2[0][1].scatter(x_testing_[:, 1], y_testing_[
                          :, 1], label='Testing data')
        ax2[0][1].scatter(x_training_[:, 1], y_training_[:, 1],
                          label='Training data')
        fitting_w_201, fitting_b_201 = linear_fitting(
            x_testing_[:, 1], y_testing_[:, 1])
        ax2[0][1].plot(x_Al_2, fitting_w_201 * x_Al_2 + fitting_b_201, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_201, fitting_w_201))
        ax2[0][1].legend(loc='upper left')
        # Si/YS
        ax2[1][0].set_xlabel('Si with regularization')
        ax2[1][0].set_ylabel('YS / MPa')
        ax2[1][0].set_title('Si / YS', fontstyle='oblique')
        ax2[1][0].scatter(x_testing_[:, 2], y_testing_[
                          :, 1], label='Testing data')
        ax2[1][0].scatter(x_training_[:, 2], y_training_[:, 1],
                          label='Training data')
        fitting_w_210, fitting_b_210 = linear_fitting(
            x_testing_[:, 2], y_testing_[:, 1])
        ax2[1][0].plot(x_Si, fitting_w_210 * x_Si + fitting_b_210, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_210, fitting_w_210))
        ax2[1][0].legend(loc='upper left')
        # AlSc2Si2/YS
        ax2[1][1].set_xlabel('AlSc2Si2 with regularization')
        ax2[1][1].set_ylabel('YS / MPa')
        ax2[1][1].set_title('AlSc2Si2 / YS', fontstyle='oblique')
        ax2[1][1].scatter(x_testing_[:, 3], y_testing_[
                          :, 1], label='Testing data')
        ax2[1][1].scatter(x_training_[:, 3], y_training_[:, 1],
                          label='Training data')
        fitting_w_211, fitting_b_211 = linear_fitting(
            x_testing_[:, 3], y_testing_[:, 1])
        ax2[1][1].plot(x_AlSc2Si2, fitting_w_211 * x_AlSc2Si2 + fitting_b_211, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_211, fitting_w_211))
        ax2[1][1].legend(loc='upper left')
        plt.savefig(path + 'phase_YS_phaseRE.png', bbox_inches='tight')
        # EL
        fig3, ax3 = plt.subplots(2, 2, figsize=(16, 12))
        # Al_1/EL
        ax3[0][0].set_xlabel('Al_1 with regularization')
        ax3[0][0].set_ylabel('EL / %')
        ax3[0][0].set_title('Al_1 / EL', fontstyle='oblique')
        ax3[0][0].scatter(x_testing_[:, 0], y_testing_[
                          :, 2], label='Testing data')
        ax3[0][0].scatter(x_training_[:, 0], y_training_[:, 2],
                          label='Training data')
        fitting_w_300, fitting_b_300 = linear_fitting(
            x_testing_[:, 0], y_testing_[:, 2])
        ax3[0][0].plot(x_Al_1, fitting_w_300 * x_Al_1 + fitting_b_300, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_300, fitting_w_300))
        ax3[0][0].legend()
        # Al_2/EL
        ax3[0][1].set_xlabel('Al_2 with regularization')
        ax3[0][1].set_ylabel('EL / %')
        ax3[0][1].set_title('Al_2 / EL', fontstyle='oblique')
        ax3[0][1].scatter(x_testing_[:, 1], y_testing_[
                          :, 2], label='Testing data')
        ax3[0][1].scatter(x_training_[:, 1], y_training_[:, 2],
                          label='Training data')
        fitting_w_301, fitting_b_301 = linear_fitting(
            x_testing_[:, 1], y_testing_[:, 2])
        ax3[0][1].plot(x_Al_2, fitting_w_301 * x_Al_2 + fitting_b_301, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_301, fitting_w_301))
        ax3[0][1].legend()
        # Si/EL
        ax3[1][0].set_xlabel('Si with regularization')
        ax3[1][0].set_ylabel('EL / %')
        ax3[1][0].set_title('Si / EL', fontstyle='oblique')
        ax3[1][0].scatter(x_testing_[:, 2], y_testing_[
                          :, 2], label='Testing data')
        ax3[1][0].scatter(x_training_[:, 2], y_training_[:, 2],
                          label='Training data')
        fitting_w_310, fitting_b_310 = linear_fitting(
            x_testing_[:, 2], y_testing_[:, 2])
        ax3[1][0].plot(x_Si, fitting_w_310 * x_Si + fitting_b_310, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_310, fitting_w_310))
        ax3[1][0].legend()
        # AlSc2Si2/EL
        ax3[1][1].set_xlabel('AlSc2Si2 with regularization')
        ax3[1][1].set_ylabel('EL / %')
        ax3[1][1].set_title('AlSc2Si2 / EL', fontstyle='oblique')
        ax3[1][1].scatter(x_testing_[:, 3], y_testing_[
                          :, 2], label='Testing data')
        ax3[1][1].scatter(x_training_[:, 3], y_training_[:, 2],
                          label='Training data')
        fitting_w_311, fitting_b_311 = linear_fitting(
            x_testing_[:, 3], y_testing_[:, 2])
        ax3[1][1].plot(x_AlSc2Si2, fitting_w_311 * x_AlSc2Si2 + fitting_b_311, color='red',
                       linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_311, fitting_w_311))
        ax3[1][1].legend()
        linear_coef_allRE = pd.DataFrame(data=np.ones((12, 2)),
                                         index=['UTS_Al_1', 'UTS_Al_2', 'UTS_Si', 'UTS_AlSc2Si2', 'YS_Al_1', 'YS_Al_2', 'YS_Si', 'YS_AlSc2Si2', 'EL_Al_1', 'EL_Al_2', 'EL_Si', 'EL_AlSc2Si2'], columns=['weight', 'bias'])
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
            path + 'linear_coef_phaseRE.csv', float_format='%.2f')
        plt.savefig(path + 'phase_EL_phaseRE.png', bbox_inches='tight')
        # plt.show()

    # 绘制相分数-性能(归一化)关系图

    def draw_relation_performanceRE(x_training_, y_training_, x_testing_, y_testing_):
        sns.set(font="Times New Roman", font_scale=1)
        x_Al_1 = np.linspace(0, 0.8, 100)
        x_Al_2_Si = np.linspace(0.1, 0.9, 100)
        x_AlSc2Si2 = np.linspace(0, 0.02, 100)
        # UTS
        fig1, ax1 = plt.subplots(1, 3, figsize=(18, 4.5))
        # Al_1/UTS
        ax1[0].set_xlabel('Al_1 / wt.%')
        ax1[0].set_ylabel('UTS with regularization')
        ax1[0].set_title('Al_1 / UTS', fontstyle='oblique')
        ax1[0].scatter(x_testing_[:, 0], y_testing_[:, 0],
                       label='Testing data')
        ax1[0].scatter(x_training_[:, 0], y_training_[:, 0],
                       label='Training data')
        fitting_w_10, fitting_b_10 = linear_fitting(x_testing_[:, 0],
                                                    y_testing_[:, 0])
        ax1[0].plot(x_Al_1, fitting_w_10 * x_Al_1 + fitting_b_10, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_10, fitting_w_10))
        ax1[0].legend()
        # Al_2/UTS
        ax1[1].set_xlabel('Al_2 + Si / wt.%')
        ax1[1].set_ylabel('UTS with regularization')
        ax1[1].set_title('Al_2 + Si / UTS', fontstyle='oblique')
        ax1[1].scatter(x_testing_[:, 1], y_testing_[:, 0],
                       label='Testing data')
        ax1[1].scatter(x_training_[:, 1], y_training_[:, 0],
                       label='Training data')
        fitting_w_11, fitting_b_11 = linear_fitting(x_testing_[:, 1],
                                                    y_testing_[:, 0])
        ax1[1].plot(x_Al_2_Si, fitting_w_11 * x_Al_2_Si + fitting_b_11, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_11, fitting_w_11))
        ax1[1].legend()
        # AlSc2Si2/UTS
        ax1[2].set_xlabel('AlSc2Si2 / wt.%')
        ax1[2].set_ylabel('UTS with regularization')
        ax1[2].set_title('AlSc2Si2 / UTS', fontstyle='oblique')
        ax1[2].scatter(x_testing_[:, 2], y_testing_[:, 0],
                       label='Testing data')
        ax1[2].scatter(x_training_[:, 2], y_training_[:, 0],
                       label='Training data')
        fitting_w_12, fitting_b_12 = linear_fitting(x_testing_[:, 2],
                                                    y_testing_[:, 0])
        ax1[2].plot(x_AlSc2Si2, fitting_w_12 * x_AlSc2Si2 + fitting_b_12, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_12, fitting_w_12))
        ax1[2].legend()
        plt.savefig(path + 'phase_UTS_performanceRE.png', bbox_inches='tight')
        # YS
        fig2, ax2 = plt.subplots(1, 3, figsize=(18, 4.5))
        # Al_1/YS
        ax2[0].set_xlabel('Al_1 / wt.%')
        ax2[0].set_ylabel('YS with regularization')
        ax2[0].set_title('Al_1 / YS', fontstyle='oblique')
        ax2[0].scatter(x_testing_[:, 0], y_testing_[:, 1],
                       label='Testing data')
        ax2[0].scatter(x_training_[:, 0], y_training_[:, 1],
                       label='Training data')
        fitting_w_20, fitting_b_20 = linear_fitting(x_testing_[:, 0],
                                                    y_testing_[:, 1])
        ax2[0].plot(x_Al_1, fitting_w_20 * x_Al_1 + fitting_b_20, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_20, fitting_w_20))
        ax2[0].legend(loc='upper left')
        # Al_2/YS
        ax2[1].set_xlabel('Al_2 + Si / wt.%')
        ax2[1].set_ylabel('YS with regularization')
        ax2[1].set_title('Al_2 / YS', fontstyle='oblique')
        ax2[1].scatter(x_testing_[:, 1], y_testing_[:, 1],
                       label='Testing data')
        ax2[1].scatter(x_training_[:, 1], y_training_[:, 1],
                       label='Training data')
        fitting_w_21, fitting_b_21 = linear_fitting(x_testing_[:, 1],
                                                    y_testing_[:, 1])
        ax2[1].plot(x_Al_2_Si, fitting_w_21 * x_Al_2_Si + fitting_b_21, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_21, fitting_w_21))
        ax2[1].legend(loc='upper left')
        # AlSc2Si2/YS
        ax2[2].set_xlabel('AlSc2Si2 / wt.%')
        ax2[2].set_ylabel('YS with regularization')
        ax2[2].set_title('AlSc2Si2 / YS', fontstyle='oblique')
        ax2[2].scatter(x_testing_[:, 2], y_testing_[:, 1],
                       label='Testing data')
        ax2[2].scatter(x_training_[:, 2], y_training_[:, 1],
                       label='Training data')
        fitting_w_22, fitting_b_22 = linear_fitting(x_testing_[:, 2],
                                                    y_testing_[:, 1])
        ax2[2].plot(x_AlSc2Si2, fitting_w_22 * x_AlSc2Si2 + fitting_b_22, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_22, fitting_w_22))
        ax2[2].legend(loc='upper left')
        plt.savefig(path + 'phase_YS_performanceRE.png', bbox_inches='tight')
        # EL
        fig3, ax3 = plt.subplots(1, 3, figsize=(18, 4.5))
        # Al_1/EL
        ax3[0].set_xlabel('Al_1 / wt.%')
        ax3[0].set_ylabel('EL with regularization')
        ax3[0].set_title('Al_1 / EL', fontstyle='oblique')
        ax3[0].scatter(x_testing_[:, 0], y_testing_[:, 2],
                       label='Testing data')
        ax3[0].scatter(x_training_[:, 0], y_training_[:, 2],
                       label='Training data')
        fitting_w_30, fitting_b_30 = linear_fitting(x_testing_[:, 0],
                                                    y_testing_[:, 2])
        ax3[0].plot(x_Al_1, fitting_w_30 * x_Al_1 + fitting_b_30, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_30, fitting_w_30))
        ax3[0].legend()
        # Al_2/EL
        ax3[1].set_xlabel('Al_2 + Si / wt.%')
        ax3[1].set_ylabel('EL with regularization')
        ax3[1].set_title('Al_2 / EL', fontstyle='oblique')
        ax3[1].scatter(x_testing_[:, 1], y_testing_[:, 2],
                       label='Testing data')
        ax3[1].scatter(x_training_[:, 1], y_training_[:, 2],
                       label='Training data')
        fitting_w_31, fitting_b_31 = linear_fitting(x_testing_[:, 1],
                                                    y_testing_[:, 2])
        ax3[1].plot(x_Al_2_Si, fitting_w_31 * x_Al_2_Si + fitting_b_31, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_31, fitting_w_31))
        ax3[1].legend()
        # AlSc2Si2/EL
        ax3[2].set_xlabel('AlSc2Si2 / wt.%')
        ax3[2].set_ylabel('EL with regularization')
        ax3[2].set_title('AlSc2Si2 / EL', fontstyle='oblique')
        ax3[2].scatter(x_testing_[:, 2], y_testing_[:, 2],
                       label='Testing data')
        ax3[2].scatter(x_training_[:, 2], y_training_[:, 2],
                       label='Training data')
        fitting_w_32, fitting_b_32 = linear_fitting(x_testing_[:, 2],
                                                    y_testing_[:, 2])
        ax3[2].plot(x_AlSc2Si2, fitting_w_32 * x_AlSc2Si2 + fitting_b_32, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_32, fitting_w_32))
        ax3[2].legend()
        plt.savefig(path + 'phase_EL_performanceRE.png', bbox_inches='tight')
        linear_coef_allRE = pd.DataFrame(data=np.ones((9, 2)),
                                         index=['UTS_Al_1', 'UTS_Al_2+Si', 'UTS_AlSc2Si2', 'YS_Al_1', 'YS_Al_2+Si', 'YS_AlSc2Si2', 'EL_Al_1', 'EL_Al_2+Si', 'EL_AlSc2Si2'], columns=['weight', 'bias'])
        # UTS
        linear_coef_allRE.iloc[0, 0] = fitting_w_10
        linear_coef_allRE.iloc[0, 1] = fitting_b_10
        linear_coef_allRE.iloc[1, 0] = fitting_w_11
        linear_coef_allRE.iloc[1, 1] = fitting_b_11
        linear_coef_allRE.iloc[2, 0] = fitting_w_12
        linear_coef_allRE.iloc[2, 1] = fitting_b_12
        # YS
        linear_coef_allRE.iloc[3, 0] = fitting_w_20
        linear_coef_allRE.iloc[3, 1] = fitting_b_20
        linear_coef_allRE.iloc[4, 0] = fitting_w_21
        linear_coef_allRE.iloc[4, 1] = fitting_b_21
        linear_coef_allRE.iloc[5, 0] = fitting_w_22
        linear_coef_allRE.iloc[5, 1] = fitting_b_22
        # EL
        linear_coef_allRE.iloc[6, 0] = fitting_w_30
        linear_coef_allRE.iloc[6, 1] = fitting_b_30
        linear_coef_allRE.iloc[7, 0] = fitting_w_31
        linear_coef_allRE.iloc[7, 1] = fitting_b_31
        linear_coef_allRE.iloc[8, 0] = fitting_w_32
        linear_coef_allRE.iloc[8, 1] = fitting_b_32
        linear_coef_allRE.to_csv(path + 'linear_coef_performanceRE.csv',
                                 float_format='%.2f')
        # plt.show()

    # 绘制相分数(归一化)-性能(归一化)关系图

    def draw_relation_allRE(x_training_, y_training_, x_testing_, y_testing_):
        sns.set(font="Times New Roman", font_scale=1)
        x_Al_1 = np.linspace(-2.3, 2.3, 100)
        x_Al_2_Si = np.linspace(-2.3, 2.3, 100)
        x_AlSc2Si2 = np.linspace(-2.3, 2.3, 100)
        # UTS
        fig1, ax1 = plt.subplots(1, 3, figsize=(18, 4.5))
        # Al_1/UTS
        ax1[0].set_xlabel('Al_1 with regularization')
        ax1[0].set_ylabel('UTS with regularization')
        ax1[0].set_title('Al_1 / UTS', fontstyle='oblique')
        ax1[0].scatter(x_testing_[:, 0], y_testing_[
            :, 0], label='Testing data')
        ax1[0].scatter(x_training_[:, 0], y_training_[:, 0],
                       label='Training data')
        fitting_w_10, fitting_b_10 = linear_fitting(x_testing_[:, 0],
                                                    y_testing_[:, 0])
        ax1[0].plot(x_Al_1, fitting_w_10 * x_Al_1 + fitting_b_10, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_10, fitting_w_10))
        ax1[0].legend()
        # Al_2+Si/UTS
        ax1[1].set_xlabel('Al_2 + Si with regularization')
        ax1[1].set_ylabel('UTS with regularization')
        ax1[1].set_title('Al_2 + Si / UTS', fontstyle='oblique')
        ax1[1].scatter(x_testing_[:, 1], y_testing_[:, 0],
                       label='Testing data')
        ax1[1].scatter(x_training_[:, 1], y_training_[:, 0],
                       label='Training data')
        fitting_w_11, fitting_b_11 = linear_fitting(x_testing_[:, 1],
                                                    y_testing_[:, 0])
        ax1[1].plot(x_Al_2_Si, fitting_w_11 * x_Al_2_Si + fitting_b_11, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_11, fitting_w_11))
        ax1[1].legend()
        # AlSc2Si2/UTS
        ax1[2].set_xlabel('AlSc2Si2 with regularization')
        ax1[2].set_ylabel('UTS with regularization')
        ax1[2].set_title('AlSc2Si2 / UTS', fontstyle='oblique')
        ax1[2].scatter(x_testing_[:, 2], y_testing_[:, 0],
                       label='Testing data')
        ax1[2].scatter(x_training_[:, 2], y_training_[:, 0],
                       label='Training data')
        fitting_w_12, fitting_b_12 = linear_fitting(x_testing_[:, 2],
                                                    y_testing_[:, 0])
        ax1[2].plot(x_AlSc2Si2, fitting_w_12 * x_AlSc2Si2 + fitting_b_12, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_12, fitting_w_12))
        ax1[2].legend()
        plt.savefig(path + 'phase_UTS_allRE.png', bbox_inches='tight')
        # YS
        fig2, ax2 = plt.subplots(1, 3, figsize=(18, 4.5))
        # Al_1/YS
        ax2[0].set_xlabel('Al_1 with regularization')
        ax2[0].set_ylabel('YS with regularization')
        ax2[0].set_title('Al_1 / YS', fontstyle='oblique')
        ax2[0].scatter(x_testing_[:, 0], y_testing_[:, 1],
                       label='Testing data')
        ax2[0].scatter(x_training_[:, 0], y_training_[:, 1],
                       label='Training data')
        fitting_w_20, fitting_b_20 = linear_fitting(x_testing_[:, 0],
                                                    y_testing_[:, 1])
        ax2[0].plot(x_Al_1, fitting_w_20 * x_Al_1 + fitting_b_20, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_20, fitting_w_20))
        ax2[0].legend(loc='upper left')
        # Al_2+Si/YS
        ax2[1].set_xlabel('Al_2 + Si with regularization')
        ax2[1].set_ylabel('YS with regularization')
        ax2[1].set_title('Al_2 + Si / YS', fontstyle='oblique')
        ax2[1].scatter(x_testing_[:, 1], y_testing_[:, 1],
                       label='Testing data')
        ax2[1].scatter(x_training_[:, 1], y_training_[:, 1],
                       label='Training data')
        fitting_w_21, fitting_b_21 = linear_fitting(x_testing_[:, 1],
                                                    y_testing_[:, 1])
        ax2[1].plot(x_Al_2_Si, fitting_w_21 * x_Al_2_Si + fitting_b_21, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_21, fitting_w_21))
        ax2[1].legend(loc='upper left')
        # AlSc2Si2/YS
        ax2[2].set_xlabel('AlSc2Si2 with regularization')
        ax2[2].set_ylabel('YS with regularization')
        ax2[2].set_title('AlSc2Si2 / YS', fontstyle='oblique')
        ax2[2].scatter(x_testing_[:, 2], y_testing_[:, 1],
                       label='Testing data')
        ax2[2].scatter(x_training_[:, 2], y_training_[:, 1],
                       label='Training data')
        fitting_w_22, fitting_b_22 = linear_fitting(x_testing_[:, 2],
                                                    y_testing_[:, 1])
        ax2[2].plot(x_AlSc2Si2, fitting_w_22 * x_AlSc2Si2 + fitting_b_22, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_22, fitting_w_22))
        ax2[2].legend(loc='upper left')
        plt.savefig(path + 'phase_YS_allRE.png', bbox_inches='tight')
        # EL
        fig3, ax3 = plt.subplots(1, 3, figsize=(18, 4.5))
        # Al_1/EL
        ax3[0].set_xlabel('Al_1 with regularization')
        ax3[0].set_ylabel('EL with regularization')
        ax3[0].set_title('Al_1 / EL', fontstyle='oblique')
        ax3[0].scatter(x_testing_[:, 0], y_testing_[:, 2],
                       label='Testing data')
        ax3[0].scatter(x_training_[:, 0], y_training_[:, 2],
                       label='Training data')
        fitting_w_30, fitting_b_30 = linear_fitting(x_testing_[:, 0],
                                                    y_testing_[:, 2])
        ax3[0].plot(x_Al_1, fitting_w_30 * x_Al_1 + fitting_b_30, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_30, fitting_w_30))
        ax3[0].legend()
        # Al_2/EL
        ax3[1].set_xlabel('Al_2 + Si with regularization')
        ax3[1].set_ylabel('EL with regularization')
        ax3[1].set_title('Al_2 + Si / EL', fontstyle='oblique')
        ax3[1].scatter(x_testing_[:, 1], y_testing_[:, 2],
                       label='Testing data')
        ax3[1].scatter(x_training_[:, 1], y_training_[:, 2],
                       label='Training data')
        fitting_w_31, fitting_b_31 = linear_fitting(x_testing_[:, 1],
                                                    y_testing_[:, 2])
        ax3[1].plot(x_Al_2_Si, fitting_w_31 * x_Al_2_Si + fitting_b_31, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_31, fitting_w_31))
        ax3[1].legend()
        # AlSc2Si2/EL
        ax3[2].set_xlabel('AlSc2Si2 with regularization')
        ax3[2].set_ylabel('EL with regularization')
        ax3[2].set_title('AlSc2Si2 / EL', fontstyle='oblique')
        ax3[2].scatter(x_testing_[:, 2], y_testing_[:, 2],
                       label='Testing data')
        ax3[2].scatter(x_training_[:, 2], y_training_[:, 2],
                       label='Training data')
        fitting_w_32, fitting_b_32 = linear_fitting(x_testing_[:, 2],
                                                    y_testing_[:, 2])
        ax3[2].plot(x_AlSc2Si2, fitting_w_32 * x_AlSc2Si2 + fitting_b_32, color='red',
                    linestyle='dashed', label='y = %.2f + (%.2f)x' % (fitting_b_32, fitting_w_32))
        ax3[2].legend()
        plt.savefig(path + 'phase_EL_allRE.png', bbox_inches='tight')
        linear_coef_allRE = pd.DataFrame(data=np.ones((9, 2)),
                                         index=['UTS_Al_1', 'UTS_Al_2+Si', 'UTS_AlSc2Si2', 'YS_Al_1', 'YS_Al_2+Si', 'YS_AlSc2Si2', 'EL_Al_1', 'EL_Al_2+Si', 'EL_AlSc2Si2'], columns=['weight', 'bias'])
        # UTS
        linear_coef_allRE.iloc[0, 0] = fitting_w_10
        linear_coef_allRE.iloc[0, 1] = fitting_b_10
        linear_coef_allRE.iloc[1, 0] = fitting_w_11
        linear_coef_allRE.iloc[1, 1] = fitting_b_11
        linear_coef_allRE.iloc[2, 0] = fitting_w_12
        linear_coef_allRE.iloc[2, 1] = fitting_b_12
        # YS
        linear_coef_allRE.iloc[3, 0] = fitting_w_20
        linear_coef_allRE.iloc[3, 1] = fitting_b_20
        linear_coef_allRE.iloc[4, 0] = fitting_w_21
        linear_coef_allRE.iloc[4, 1] = fitting_b_21
        linear_coef_allRE.iloc[5, 0] = fitting_w_22
        linear_coef_allRE.iloc[5, 1] = fitting_b_22
        # EL
        linear_coef_allRE.iloc[6, 0] = fitting_w_30
        linear_coef_allRE.iloc[6, 1] = fitting_b_30
        linear_coef_allRE.iloc[7, 0] = fitting_w_31
        linear_coef_allRE.iloc[7, 1] = fitting_b_31
        linear_coef_allRE.iloc[8, 0] = fitting_w_32
        linear_coef_allRE.iloc[8, 1] = fitting_b_32
        linear_coef_allRE.to_csv(path + 'linear_coef_allRE.csv',
                                 float_format='%.2f')
        # plt.show()

    # 获取数据

    x, y_UTS, y_YS, y_EL, EL_Si, EL_Mg = get_training_data(
        training_data_file_path)
    x_testing, EL_Si_test, EL_Mg_test = get_testing_data(
        testing_data_file_path)

    # 执行正则化，并记住训练集数据的正则化规则,运用于测试集数据

    x_scaler = StandardScaler().fit(x)
    x_standarded = torch.from_numpy(x_scaler.transform(x)).float()
    x_standarded_test = torch.from_numpy(x_scaler.transform(x_testing)).float()

    # 调用模型进行预测

    y_list = torch.cat((y_UTS, y_YS, y_EL), 1)
    with torch.no_grad():
        y_testing = test(model_path, x_standarded_test)
        y_scaler = StandardScaler().fit(y_testing)
        y_standarded = torch.from_numpy(y_scaler.transform(y_list)).float()
        y_standarded_test = torch.from_numpy(
            y_scaler.transform(y_testing)).float()
        if np.isnan(y_testing.numpy().any()):
            print('==============Prediction run out of range===============')
        else:
            print('==================Prediction complete===================')

            # 数据可视化(散点图)
            draw_scatter(EL_Si.numpy(), EL_Mg.numpy(), y_UTS.numpy(),
                         EL_Si_test.numpy(), EL_Mg_test.numpy(), y_testing.numpy()[:, 0], 'UTS / MPa')
            draw_scatter(EL_Si.numpy(), EL_Mg.numpy(), y_YS.numpy(),
                         EL_Si_test.numpy(), EL_Mg_test.numpy(), y_testing.numpy()[:, 1], 'YS / MPa')
            draw_scatter(EL_Si.numpy(), EL_Mg.numpy(), y_EL.numpy(),
                         EL_Si_test.numpy(), EL_Mg_test.numpy(), y_testing.numpy()[:, 2], 'EL / %')

            # 综合力学性能计算及可视化
            # data_process(path, EL_Si_test.numpy(), EL_Mg_test.numpy())

            # # 绘制相分数-性能关系图
            draw_relation(x.numpy(), y_list.numpy(),
                          x_testing.numpy(), y_testing.numpy())

            # # 绘制相分数-性能(正则化)关系图
            draw_relation_performanceRE(x.numpy(), y_standarded.numpy(),
                                        x_testing.numpy(), y_standarded_test.numpy())

            # # 绘制相分数(正则化)-性能关系图
            # draw_relation_phaseRE(x_standarded.numpy(), y_list.numpy(),
            #                       x_standarded_test.numpy(), y_testing.numpy())

            # 绘制相分数(正则化)-性能(正则化)关系图
            draw_relation_allRE(x_standarded.numpy(), y_standarded.numpy(),
                                x_standarded_test.numpy(), y_standarded_test.numpy())


if __name__ == '__main__':
    main(parameters_list)
