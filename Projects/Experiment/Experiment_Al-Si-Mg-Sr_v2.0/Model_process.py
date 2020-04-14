import os

import matplotlib
import matplotlib.pyplot as plt
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
    add = parameters_list[7]
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

    def draw_scatter(x_training, y_training, x_predicting, y_predicting, error, add):
        ymin1 = 75
        ymax1 = 270
        ymin2 = 0
        ymax2 = 30

        if add:
            y_index = -1
        else:
            y_index = train_end_index

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
        pl12 = ax1.errorbar(x_training[0:y_index], y_training[0:y_index, 0], yerr=error[0],
                            linestyle='None', capsize=5, ecolor='royalblue',
                            fmt='o:', mfc='wheat', mec='royalblue', ms=5)
        # YS
        pl21 = ax1.scatter(x_predicting, y_predicting[:, 1],
                           s=15, color='mediumseagreen')
        pl22 = ax1.errorbar(x_training[0:y_index], y_training[0:y_index, 1], yerr=error[1],
                            linestyle='None', capsize=5, ecolor='green',
                            fmt='o:', mfc='wheat', mec='green', ms=5)
        # EL
        pl31 = ax2.scatter(x_predicting, y_predicting[:, 2],
                           s=15, color='chocolate')
        pl32 = ax2.errorbar(x_training[0:y_index], y_training[0:y_index, 2], yerr=error[2],
                            linestyle='None', capsize=5, ecolor='saddlebrown',
                            fmt='o:', mfc='wheat', mec='saddlebrown', ms=5)

        ax1.vlines(0.005, ymin1, ymax1,
                   linestyles='dotted', linewidth=2)
        ax1.vlines(0.077, ymin1, ymax1,
                   linestyles='dotted', linewidth=2)
        # ax1.vlines(0.318, ymin1, ymax1,
        #            linestyles='dotted', linewidth=2)
        ax1.text(0.01, 130, 'w(Sr) = 0.005', fontdict={'style': 'italic'})
        ax1.text(0.08, 130, 'w(Sr) = 0.077', fontdict={'style': 'italic'})
        # ax1.text(0.32, 180, 'w(Sr) = 0.318', fontdict={'style': 'italic'})

        label11 = 'Predicted UTS'
        label12 = 'Experimental UTS'
        label21 = 'Predicted YS'
        label22 = 'Experimental YS'
        label31 = 'Predicted EL'
        label32 = 'Experimental EL'

        # Additional
        if add:
            # UTS
            x_add1 = np.array([x_training[y_index]])
            y_add1 = np.array([y_training[y_index, 0]])
            e_add1 = np.array([[3],
                               [3]])
            pl_add = ax1.errorbar(x_add1, y_add1, yerr=e_add1,
                                  linestyle='None', capsize=5, ecolor='r',
                                  fmt='*', mfc='white', mec='r', ms=10)
            # YS
            x_add2 = np.array([x_training[y_index]])
            y_add2 = np.array([y_training[y_index, 1]])
            e_add2 = np.array([[3],
                               [3]])
            ax1.errorbar(x_add2, y_add2, yerr=e_add2,
                         linestyle='None', capsize=5, ecolor='r',
                         fmt='*', mfc='white', mec='r', ms=10)
            # EL
            x_add3 = np.array([x_training[y_index]])
            y_add3 = np.array([y_training[y_index, 2]])
            e_add3 = np.array([[1],
                               [1]])
            ax2.errorbar(x_add3, y_add3, yerr=e_add3,
                         linestyle='None', capsize=5, ecolor='r',
                         fmt='*', mfc='white', mec='r', ms=10)
            label_add = 'Additional experiment'

            plt.legend([pl11, pl21, pl31, pl_add, pl12, pl22, pl32],
                       [label11, label21, label31, label_add,
                        label12, label22, label32],
                       loc='upper right', frameon=False, ncol=2)
        else:
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
        Al_line = False
        Al2Si2Sr_line = False

        if item == 'Al':
            item_name = 'Al phase'
            x_phase = np.linspace(-1.5, 1, 100)
            x_index11 = 0
            x_index12 = 154
            x_index2 = 0
            Al_line = True
        elif item == 'Si':
            item_name = 'Si phase'
            x_phase = np.linspace(-1, 3, 100)
            x_index11 = 6
            x_index12 = 154
            x_index2 = 1
            y_max = 7
        else:
            item_name = 'Al${_2}$Si${_2}$Sr phase'
            x_phase = np.linspace(-1, 2, 100)
            x_index11 = 0
            x_index12 = 154
            x_index2 = 2
            y_max = 7
            Al2Si2Sr_line = True

        xlabel = 'Phase fraction of %s with regularization' % item_name
        ylabel = 'Performance with regularization'

        fig = plt.figure(figsize=(8, 6))
        ax = plt.subplot()
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # if Al_line:
        #     ax.vlines(0.87, -6, 2, linestyles='dotted', linewidth=2)
        #     ax.vlines(-0.89, -6, 2, linestyles='dotted', linewidth=2)
        #     ax.vlines(-1.37, -6, 2, linestyles='dotted', linewidth=2)
        #     ax.text(0.3, -3, 'w(Sr) = 0.005',
        #             fontdict={'style': 'italic'})
        #     ax.text(-0.84, -4, 'w(Sr) = 0.06',
        #             fontdict={'style': 'italic'})
        #     ax.text(-1.32, -3, 'w(Sr) = 0.075',
        #             fontdict={'style': 'italic'})
        # if Al2Si2Sr_line:
        #     ax.vlines(-0.8, -6, 4, linestyles='dotted', linewidth=2)
        #     ax.vlines(0.35, -6, 4, linestyles='dotted', linewidth=2)
        #     ax.vlines(0.67, -6, 4, linestyles='dotted', linewidth=2)
        #     ax.text(-0.75, -3, 'w(Sr) = 0.005',
        #             fontdict={'style': 'italic'})
        #     ax.text(-0.3, -4, 'w(Sr) = 0.06',
        #             fontdict={'style': 'italic'})
        #     ax.text(0.72, -3, 'w(Sr) = 0.075',
        #             fontdict={'style': 'italic'})

        # UTS
        pl11 = ax.scatter(x_testing_[:, x_index2], y_testing_[:, 0], s=15)
        fitting_w1, fitting_b1 = linear_fitting(x_testing_[x_index11:x_index12, x_index2],
                                                y_testing_[x_index11:x_index12, 0])
        pl12, = ax.plot(x_phase, fitting_w1 * x_phase +
                        fitting_b1, linestyle='dashed', linewidth=2)
        # YS
        pl21 = ax.scatter(x_testing_[:, x_index2], y_testing_[:, 1], s=15)
        fitting_w2, fitting_b2 = linear_fitting(x_testing_[x_index11:x_index12, x_index2],
                                                y_testing_[x_index11:x_index12, 1])
        pl22, = ax.plot(x_phase, fitting_w2 * x_phase +
                        fitting_b2, linestyle='dashed', linewidth=2)
        # EL
        pl31 = ax.scatter(x_testing_[:, x_index2], y_testing_[:, 2], s=15)
        fitting_w3, fitting_b3 = linear_fitting(x_testing_[x_index11:x_index12, x_index2],
                                                y_testing_[x_index11:x_index12, 2])
        pl32, = ax.plot(x_phase, fitting_w3 * x_phase +
                        fitting_b3, linestyle='dashed', linewidth=2)

        label11 = 'UTS'
        label12 = 'w${_{UTS}}$ = %.2f  b${_{UTS}}$ = %.2f' % (
            fitting_w1, fitting_b1)
        label21 = 'YS'
        label22 = 'w${_{YS}}$ = %.2f  b${_{YS}}$ = %.2f' % (
            fitting_w2, fitting_b2)
        label31 = 'EL'
        label32 = 'w${_{EL}}$ = %.2f  b${_{EL}}$ = %.2f' % (
            fitting_w3, fitting_b3)

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

    def draw_relation_allRE_whole(x_training_, y_training_, x_testing_, y_testing_):
        sns.set(font="Times New Roman", font_scale=1.3, style='ticks')
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'
        x_Al = np.linspace(-1.5, 1, 100)
        x_Si = np.linspace(-1, 3, 100)
        x_AlSi2Sr = np.linspace(-1, 2.2, 100)
        # UTS
        # fig1, ax1 = plt.subplots(2, 2, figsize=(16, 12))
        fig = plt.figure(figsize=(24, 18))
        ax1 = plt.subplot(331)
        ax2 = plt.subplot(332)
        ax3 = plt.subplot(333)
        ax4 = plt.subplot(334)
        ax5 = plt.subplot(335)
        ax6 = plt.subplot(336)
        ax7 = plt.subplot(337)
        ax8 = plt.subplot(338)
        ax9 = plt.subplot(339)
        ax1.set_ylim(-4, 5)
        ax2.set_ylim(-4, 5)
        ax3.set_ylim(-4, 5)
        ax4.set_ylim(-2, 5)
        ax5.set_ylim(-2, 5)
        ax6.set_ylim(-2, 5)
        ax7.set_ylim(-8, 6)
        ax8.set_ylim(-8, 6)
        ax9.set_ylim(-8, 6)
        ax1.spines['left'].set_color('cornflowerblue')
        ax1.tick_params(axis='y', colors='cornflowerblue')
        ax2.spines['left'].set_color('cornflowerblue')
        ax2.tick_params(axis='y', colors='cornflowerblue')
        ax3.spines['left'].set_color('cornflowerblue')
        ax3.tick_params(axis='y', colors='cornflowerblue')
        ax4.spines['left'].set_color('chocolate')
        ax4.tick_params(axis='y', colors='chocolate')
        ax5.spines['left'].set_color('chocolate')
        ax5.tick_params(axis='y', colors='chocolate')
        ax6.spines['left'].set_color('chocolate')
        ax6.tick_params(axis='y', colors='chocolate')
        ax7.spines['left'].set_color('mediumseagreen')
        ax7.tick_params(axis='y', colors='mediumseagreen')
        ax8.spines['left'].set_color('mediumseagreen')
        ax8.tick_params(axis='y', colors='mediumseagreen')
        ax9.spines['left'].set_color('mediumseagreen')
        ax9.tick_params(axis='y', colors='mediumseagreen')
        # Al/UTS
        ax1.set_ylabel('UTS with regularization', color='cornflowerblue')
        ax1.scatter(x_testing_[:, 0], y_testing_[:, 0],
                    label='Predicted UTS', color='cornflowerblue')
        fitting_w_1, fitting_b_1 = linear_fitting(x_testing_[0:154, 0],
                                                  y_testing_[0:154, 0])
        ax1.plot(x_Al, fitting_w_1 * x_Al + fitting_b_1, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_1, fitting_b_1))
        ax1.legend(loc='upper right', frameon=False)
        # Si/UTS
        ax2.scatter(x_testing_[:, 1], y_testing_[:, 0],
                    label='Predicted UTS', color='cornflowerblue')
        fitting_w_2, fitting_b_2 = linear_fitting(x_testing_[6:154, 1],
                                                  y_testing_[6:154, 0])
        ax2.plot(x_Si, fitting_w_2 * x_Si + fitting_b_2, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_2, fitting_b_2))
        ax2.legend(loc='upper right', frameon=False)
        # AlSi2Sr/UTS
        ax3.scatter(x_testing_[:, 2], y_testing_[:, 0],
                    label='Predicted UTS', color='cornflowerblue')
        fitting_w_3, fitting_b_3 = linear_fitting(x_testing_[0:154, 2],
                                                  y_testing_[0:154, 0])
        ax3.plot(x_AlSi2Sr, fitting_w_3 * x_AlSi2Sr + fitting_b_3, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_3, fitting_b_3))
        ax3.legend(loc='upper right', frameon=False)
        # Al/YS
        ax4.set_ylabel('YS with regularization', color='chocolate')
        ax4.scatter(x_testing_[:, 0], y_testing_[:, 1],
                    label='Predicted YS', color='chocolate')
        fitting_w_4, fitting_b_4 = linear_fitting(x_testing_[0:154, 0],
                                                  y_testing_[0:154, 1])
        ax4.plot(x_Al, fitting_w_4 * x_Al + fitting_b_4, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_4, fitting_b_4))
        ax4.legend(loc='upper right', frameon=False)
        # Si/YS
        ax5.scatter(x_testing_[:, 1], y_testing_[:, 1],
                    label='Predicted YS', color='chocolate')
        fitting_w_5, fitting_b_5 = linear_fitting(x_testing_[6:154, 1],
                                                  y_testing_[6:154, 1])
        ax5.plot(x_Si, fitting_w_5 * x_Si + fitting_b_5, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_5, fitting_b_5))
        ax5.legend(loc='upper right', frameon=False)
        # AlSi2Sr/YS
        ax6.scatter(x_testing_[:, 2], y_testing_[:, 1],
                    label='Predicted YS', color='chocolate')
        fitting_w_6, fitting_b_6 = linear_fitting(x_testing_[0:154, 2],
                                                  y_testing_[0:154, 1])
        ax6.plot(x_AlSi2Sr, fitting_w_6 * x_AlSi2Sr + fitting_b_6, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_6, fitting_b_6))
        ax6.legend(loc='upper right', frameon=False)
        # Al/EL
        ax7.set_xlabel('Phase fraction of Al phase with regularization')
        ax7.set_ylabel('EL with regularization', color='mediumseagreen')
        ax7.scatter(x_testing_[:, 0], y_testing_[:, 2],
                    label='Predicted EL', color='mediumseagreen')
        fitting_w_7, fitting_b_7 = linear_fitting(x_testing_[0:154, 0],
                                                  y_testing_[0:154, 2])
        ax7.plot(x_Al, fitting_w_7 * x_Al + fitting_b_7, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_7, fitting_b_7))
        ax7.legend(loc='upper right', frameon=False)
        # Si/EL
        ax8.set_xlabel('Phase fraction of Si phase with regularization')
        ax8.scatter(x_testing_[:, 1], y_testing_[:, 2],
                    label='Predicted EL', color='mediumseagreen')
        fitting_w_8, fitting_b_8 = linear_fitting(x_testing_[6:154, 1],
                                                  y_testing_[6:154, 2])
        ax8.plot(x_Si, fitting_w_8 * x_Si + fitting_b_8, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_8, fitting_b_8))
        ax8.legend(loc='upper right', frameon=False)
        # AlSi2Sr/EL
        ax9.set_xlabel(
            'Phase fraction of Al${_2}$Si${_2}$Sr phase with regularization')
        ax9.scatter(x_testing_[:, 2], y_testing_[:, 2],
                    label='Predicted EL', color='mediumseagreen')
        fitting_w_9, fitting_b_9 = linear_fitting(x_testing_[0:154, 2],
                                                  y_testing_[0:154, 2])
        ax9.plot(x_AlSi2Sr, fitting_w_9 * x_AlSi2Sr + fitting_b_9, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_9, fitting_b_9))
        ax9.legend(loc='upper right', frameon=False)
        plt.savefig(path + 'phase_allRE.png', bbox_inches='tight')
        linear_coef_allRE = pd.DataFrame(data=np.ones((9, 2)),
                                         index=['UTS_Al', 'UTS_Si', 'UTS_AlSi2Sr',
                                                'YS_Al', 'YS_Si', 'YS_AlSi2Sr',
                                                'EL_Al', 'EL_Si', 'EL_AlSi2Sr'],
                                         columns=['weight', 'bias'])
        # UTS
        linear_coef_allRE.iloc[0, 0] = fitting_w_1
        linear_coef_allRE.iloc[0, 1] = fitting_b_1
        linear_coef_allRE.iloc[1, 0] = fitting_w_2
        linear_coef_allRE.iloc[1, 1] = fitting_b_2
        linear_coef_allRE.iloc[2, 0] = fitting_w_3
        linear_coef_allRE.iloc[2, 1] = fitting_b_3
        # YS
        linear_coef_allRE.iloc[3, 0] = fitting_w_4
        linear_coef_allRE.iloc[3, 1] = fitting_b_4
        linear_coef_allRE.iloc[4, 0] = fitting_w_5
        linear_coef_allRE.iloc[4, 1] = fitting_b_5
        linear_coef_allRE.iloc[5, 0] = fitting_w_6
        linear_coef_allRE.iloc[5, 1] = fitting_b_6
        # EL
        linear_coef_allRE.iloc[6, 0] = fitting_w_7
        linear_coef_allRE.iloc[6, 1] = fitting_b_7
        linear_coef_allRE.iloc[7, 0] = fitting_w_8
        linear_coef_allRE.iloc[7, 1] = fitting_b_8
        linear_coef_allRE.iloc[8, 0] = fitting_w_9
        linear_coef_allRE.iloc[8, 1] = fitting_b_9
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
    xRe = pd.DataFrame(data=x_standarded_predict.numpy(),
                       columns=['Al', 'Si', 'Al2Si2Sr'])
    xRe.to_csv(path + 'xRe.csv', float_format='%.2f')

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
        # print(x_predicting[:, 0].min())
        # print(x_predicting[:, 0].max())
        # print(x_predicting[:, 1].min())
        # print(x_predicting[:, 1].max())
        # print(x_predicting[:, 2].min())
        # print(x_predicting[:, 2].max())
        # print(EL_Sr.numpy().T[0])
        # print(y_standarded.numpy()[:, 0])
        if np.isnan(y_predicting.numpy().any()):
            print('==============Prediction run out of range===============')
        else:
            print('==================Prediction complete===================')

            # 数据可视化(散点图)
            draw_scatter(EL_Sr.numpy(), y_list.numpy(),
                         EL_Sr_predicting.numpy(), y_predicting.numpy(),
                         error, add)

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
            draw_relation_allRE_whole(x_standarded.numpy(), y_standarded.numpy(),
                                      x_standarded_predict.numpy(), y_standarded_predict.numpy())


if __name__ == '__main__':
    main(parameters_list)
