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
        x = torch.from_numpy(data.loc[:, 'PH_Al':'PH_AlSc2Si2'].values).float()
        EL_Sc = torch.unsqueeze(
            (torch.from_numpy(data['EL_Sc'].values)), dim=1).float()
        return x, EL_Sc

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
        EL_Sc = torch.unsqueeze(
            (torch.from_numpy(data['EL_Sc'].values)), dim=1).float()
        return x, y_UTS, y_YS, y_EL, EL_Sc

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

    # R value

    def correlation_coefficient(real, predict):
        real_mean = real.mean()
        predict_mean = predict.mean()
        SSR = 0
        r = 0
        p = 0
        for i in range(len(real)):
            SSR += (real[i] - real_mean) * (predict[i] - predict_mean)
            r += (real[i] - real_mean) ** 2
            p += (predict[i] - predict_mean) ** 2
        SST = np.sqrt(r * p)
        return SSR / SST

    # 绘制散点图

    def draw_scatter(x_training, y_training, x_predict, y_predict, error, add, R_value):
        ymin1 = 90
        ymax1 = 300
        ymin2 = -8
        ymax2 = 20

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
        ax1.set_xlabel('Sc / wt. %')
        ax1.set_ylabel('Strength / MPa')
        ax1.set_ylim(ymin1, ymax1)
        ax2.set_ylabel('Elongation / %', color='chocolate')
        ax2.spines['right'].set_color('chocolate')
        ax2.yaxis.label.set_color('chocolate')
        ax2.tick_params(axis='y', colors='chocolate')
        ax2.set_ylim(ymin2, ymax2)

        # UTS
        pl11 = ax1.scatter(x_predict, y_predict[:, 0],
                           s=15, color='cornflowerblue')
        pl12 = ax1.errorbar(x_training[0:y_index], y_training[0:y_index, 0], yerr=error[0],
                            linestyle='None', capsize=5, ecolor='royalblue',
                            fmt='o:', mfc='wheat', mec='royalblue', ms=5)
        # YS
        pl21 = ax1.scatter(x_predict, y_predict[:, 1],
                           s=15, color='mediumseagreen')
        pl22 = ax1.errorbar(x_training[0:y_index], y_training[0:y_index, 1], yerr=error[1],
                            linestyle='None', capsize=5, ecolor='green',
                            fmt='o:', mfc='wheat', mec='green', ms=5)
        # EL
        pl31 = ax2.scatter(x_predict, y_predict[:, 2],
                           s=15, color='chocolate')
        pl32 = ax2.errorbar(x_training[0:y_index], y_training[0:y_index, 2], yerr=error[2],
                            linestyle='None', capsize=5, ecolor='saddlebrown',
                            fmt='o:', mfc='wheat', mec='saddlebrown', ms=5)

        ax1.vlines(0.54, ymin1, ymax1,
                   linestyles='dotted', linewidth=2)
        ax1.text(0.56, 118, 'w(Sc) = 0.54', fontdict={'style': 'italic'})

        ax1.set_title('R = %.6f' % R_value, y=1.015, x=0.8,
                      fontdict={'style': 'oblique', 'color': 'r'})

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

    def draw_relation_allRE(x_predict, y_predict, item):
        sns.set(font="Times New Roman", font_scale=1.3, style='ticks')
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'
        y_min = -7
        y_max = 5

        if item == 'Al':
            item_name = r'$\alpha$-(Al) phase'
            x_phase = np.linspace(-2, 1, 100)
            x_index11 = 0
            x_index12 = 750
            x_index2 = 0
        elif item == 'Al2':
            item_name = 'Eutectic (Al) phase'
            x_phase = np.linspace(-1.3, 1.9, 100)
            x_index11 = 0
            x_index12 = 750
            x_index2 = 1
            y_max = 7
        elif item == 'Si':
            item_name = 'Eutectic (Si) phase'
            x_phase = np.linspace(-1.8, 2.1, 100)
            x_index11 = 0
            x_index12 = 750
            x_index2 = 2
            y_max = 7
        else:
            item_name = 'AlSc${_2}$Si${_2}$ phase'
            x_phase = np.linspace(-1.7, 1.3, 100)
            x_index11 = 0
            x_index12 = 750
            x_index2 = 3
            y_max = 7

        xlabel = 'Phase fraction of %s with regularization' % item_name
        ylabel = 'Performance with regularization'

        fig = plt.figure(figsize=(8, 6))
        ax = plt.subplot()
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # UTS
        pl11 = ax.scatter(x_predict[:, x_index2], y_predict[:, 0], s=15)
        fitting_w1, fitting_b1 = linear_fitting(x_predict[x_index11:x_index12, x_index2],
                                                y_predict[x_index11:x_index12, 0])
        pl12, = ax.plot(x_phase, fitting_w1 * x_phase +
                        fitting_b1, linestyle='dashed', linewidth=2)
        # YS
        pl21 = ax.scatter(x_predict[:, x_index2], y_predict[:, 1], s=15)
        fitting_w2, fitting_b2 = linear_fitting(x_predict[x_index11:x_index12, x_index2],
                                                y_predict[x_index11:x_index12, 1])
        pl22, = ax.plot(x_phase, fitting_w2 * x_phase +
                        fitting_b2, linestyle='dashed', linewidth=2)
        # EL
        pl31 = ax.scatter(x_predict[:, x_index2], y_predict[:, 2], s=15)
        fitting_w3, fitting_b3 = linear_fitting(x_predict[x_index11:x_index12, x_index2],
                                                y_predict[x_index11:x_index12, 2])
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

    def draw_relation_allRE_whole(x_predict, y_predict):
        sns.set(font="Times New Roman", font_scale=1.3, style='ticks')
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'
        x_Al = np.linspace(-2, 1, 100)
        x_Al2 = np.linspace(-1.3, 1.9, 100)
        x_Si = np.linspace(-1.8, 2.1, 100)
        x_AlSc2Si2 = np.linspace(-1.7, 1.3, 100)
        # 取值范围
        # 两端
        # start = 29
        # end = 540
        # 全部
        start = 0
        end = 751
        # 部分
        # start = 540
        # end = 751
        # UTS
        fig = plt.figure(figsize=(32, 18))
        ax1 = plt.subplot(3, 4, 1)
        ax2 = plt.subplot(3, 4, 2)
        ax3 = plt.subplot(3, 4, 3)
        ax4 = plt.subplot(3, 4, 4)
        ax5 = plt.subplot(3, 4, 5)
        ax6 = plt.subplot(3, 4, 6)
        ax7 = plt.subplot(3, 4, 7)
        ax8 = plt.subplot(3, 4, 8)
        ax9 = plt.subplot(3, 4, 9)
        ax10 = plt.subplot(3, 4, 10)
        ax11 = plt.subplot(3, 4, 11)
        ax12 = plt.subplot(3, 4, 12)
        ax1.set_ylim(-4, 2.5)
        ax2.set_ylim(-4, 2.5)
        ax3.set_ylim(-4, 2.5)
        ax4.set_ylim(-4, 2.5)
        ax5.set_ylim(-4, 2.5)
        ax6.set_ylim(-4, 2.5)
        ax7.set_ylim(-4, 2.5)
        ax8.set_ylim(-4, 2.5)
        ax9.set_ylim(-4, 2.5)
        ax10.set_ylim(-4, 2.5)
        ax11.set_ylim(-4, 2.5)
        ax12.set_ylim(-4, 2.5)
        ax1.spines['left'].set_color('cornflowerblue')
        ax1.tick_params(axis='y', colors='cornflowerblue')
        ax2.spines['left'].set_color('cornflowerblue')
        ax2.tick_params(axis='y', colors='cornflowerblue')
        ax3.spines['left'].set_color('cornflowerblue')
        ax3.tick_params(axis='y', colors='cornflowerblue')
        ax4.spines['left'].set_color('cornflowerblue')
        ax4.tick_params(axis='y', colors='cornflowerblue')
        ax5.spines['left'].set_color('chocolate')
        ax5.tick_params(axis='y', colors='chocolate')
        ax6.spines['left'].set_color('chocolate')
        ax6.tick_params(axis='y', colors='chocolate')
        ax7.spines['left'].set_color('chocolate')
        ax7.tick_params(axis='y', colors='chocolate')
        ax8.spines['left'].set_color('chocolate')
        ax8.tick_params(axis='y', colors='chocolate')
        ax9.spines['left'].set_color('mediumseagreen')
        ax9.tick_params(axis='y', colors='mediumseagreen')
        ax10.spines['left'].set_color('mediumseagreen')
        ax10.tick_params(axis='y', colors='mediumseagreen')
        ax11.spines['left'].set_color('mediumseagreen')
        ax11.tick_params(axis='y', colors='mediumseagreen')
        ax12.spines['left'].set_color('mediumseagreen')
        ax12.tick_params(axis='y', colors='mediumseagreen')
        # Al/UTS
        ax1.set_ylabel('UTS with regularization', color='cornflowerblue')
        ax1.scatter(x_predict[start:end, 0], y_predict[start:end, 0],
                    label='Predicted UTS', color='cornflowerblue')
        fitting_w_1, fitting_b_1 = linear_fitting(x_predict[start:end, 0],
                                                  y_predict[start:end, 0])
        ax1.plot(x_Al, fitting_w_1 * x_Al + fitting_b_1, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_1, fitting_b_1))
        ax1.legend(loc='upper right', frameon=False)
        # Al2/UTS
        ax2.set_ylabel('UTS with regularization', color='cornflowerblue')
        ax2.scatter(x_predict[start:end, 1], y_predict[start:end, 0],
                    label='Predicted UTS', color='cornflowerblue')
        fitting_w_2, fitting_b_2 = linear_fitting(x_predict[start:end, 1],
                                                  y_predict[start:end, 0])
        ax2.plot(x_Al2, fitting_w_2 * x_Al2 + fitting_b_2, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_2, fitting_b_2))
        ax2.legend(loc='upper right', frameon=False)
        # Si/UTS
        ax3.scatter(x_predict[start:end, 2], y_predict[start:end, 0],
                    label='Predicted UTS', color='cornflowerblue')
        fitting_w_3, fitting_b_3 = linear_fitting(x_predict[start:end, 2],
                                                  y_predict[start:end, 0])
        ax3.plot(x_Si, fitting_w_3 * x_Si + fitting_b_3, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_3, fitting_b_3))
        ax3.legend(loc='upper right', frameon=False)
        # AlSc2Si2/UTS
        ax4.scatter(x_predict[start:end, 3], y_predict[start:end, 0],
                    label='Predicted UTS', color='cornflowerblue')
        fitting_w_4, fitting_b_4 = linear_fitting(x_predict[start:end, 3],
                                                  y_predict[start:end, 0])
        ax4.plot(x_AlSc2Si2, fitting_w_4 * x_AlSc2Si2 + fitting_b_4, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_4, fitting_b_4))
        ax4.legend(loc='upper right', frameon=False)
        # Al/YS
        ax5.set_ylabel('YS with regularization', color='chocolate')
        ax5.scatter(x_predict[start:end, 0], y_predict[start:end, 1],
                    label='Predicted YS', color='chocolate')
        fitting_w_5, fitting_b_5 = linear_fitting(x_predict[start:end, 0],
                                                  y_predict[start:end, 1])
        ax5.plot(x_Al, fitting_w_5 * x_Al + fitting_b_5, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_5, fitting_b_5))
        ax5.legend(loc='upper right', frameon=False)
        # Al2/YS
        ax6.set_ylabel('YS with regularization', color='chocolate')
        ax6.scatter(x_predict[start:end, 1], y_predict[start:end, 1],
                    label='Predicted YS', color='chocolate')
        fitting_w_6, fitting_b_6 = linear_fitting(x_predict[start:end, 1],
                                                  y_predict[start:end, 1])
        ax6.plot(x_Al2, fitting_w_6 * x_Al2 + fitting_b_6, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_6, fitting_b_6))
        ax6.legend(loc='upper right', frameon=False)
        # Si/YS
        ax7.scatter(x_predict[start:end, 2], y_predict[start:end, 1],
                    label='Predicted YS', color='chocolate')
        fitting_w_7, fitting_b_7 = linear_fitting(x_predict[start:end, 2],
                                                  y_predict[start:end, 1])
        ax7.plot(x_Si, fitting_w_7 * x_Si + fitting_b_7, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_7, fitting_b_7))
        ax7.legend(loc='upper right', frameon=False)
        # AlSc2Si2/YS
        ax8.scatter(x_predict[start:end, 3], y_predict[start:end, 1],
                    label='Predicted YS', color='chocolate')
        fitting_w_8, fitting_b_8 = linear_fitting(x_predict[start:end, 3],
                                                  y_predict[start:end, 1])
        ax8.plot(x_AlSc2Si2, fitting_w_8 * x_AlSc2Si2 + fitting_b_8, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_8, fitting_b_8))
        ax8.legend(loc='upper right', frameon=False)
        # Al/EL
        ax9.set_xlabel(
            r'Phase fraction of $\alpha$-(Al) phase with regularization')
        ax9.set_ylabel('EL with regularization', color='mediumseagreen')
        ax9.scatter(x_predict[start:end, 0], y_predict[start:end, 2],
                    label='Predicted EL', color='mediumseagreen')
        fitting_w_9, fitting_b_9 = linear_fitting(x_predict[start:end, 0],
                                                  y_predict[start:end, 2])
        ax9.plot(x_Al, fitting_w_9 * x_Al + fitting_b_9, color='red',
                 linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_9, fitting_b_9))
        ax9.legend(loc='upper right', frameon=False)
        # Al2/EL
        ax10.set_xlabel(
            'Phase fraction of eutectic (Al) phase with regularization')
        ax10.set_ylabel('EL with regularization', color='mediumseagreen')
        ax10.scatter(x_predict[start:end, 1], y_predict[start:end, 2],
                     label='Predicted EL', color='mediumseagreen')
        fitting_w_10, fitting_b_10 = linear_fitting(x_predict[start:end, 1],
                                                    y_predict[start:end, 2])
        ax10.plot(x_Al2, fitting_w_10 * x_Al2 + fitting_b_10, color='red',
                  linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_10, fitting_b_10))
        ax10.legend(loc='upper right', frameon=False)
        # Si/EL
        ax11.set_xlabel('Phase fraction of Si phase with regularization')
        ax11.scatter(x_predict[start:end, 2], y_predict[start:end, 2],
                     label='Predicted EL', color='mediumseagreen')
        fitting_w_11, fitting_b_11 = linear_fitting(x_predict[start:end, 2],
                                                    y_predict[start:end, 2])
        ax11.plot(x_Si, fitting_w_11 * x_Si + fitting_b_11, color='red',
                  linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_11, fitting_b_11))
        ax11.legend(loc='upper right', frameon=False)
        # AlSc2Si2/EL
        ax12.set_xlabel(
            'Phase fraction of AlSc${_2}$Si${_2}$ phase with regularization')
        ax12.scatter(x_predict[start:end, 3], y_predict[start:end, 2],
                     label='Predicted EL', color='mediumseagreen')
        fitting_w_12, fitting_b_12 = linear_fitting(x_predict[start:end, 3],
                                                    y_predict[start:end, 2])
        ax12.plot(x_AlSc2Si2, fitting_w_12 * x_AlSc2Si2 + fitting_b_12, color='red',
                  linestyle='dashed', label='w = %.2f  b = %.2f' % (fitting_w_12, fitting_b_12))
        ax12.legend(loc='upper right', frameon=False)
        plt.savefig(path + 'phase_allRE.png', bbox_inches='tight')
        linear_coef_allRE = pd.DataFrame(data=np.ones((12, 2)),
                                         index=['UTS_Al', 'UTS_Al2', 'UTS_Si', 'UTS_AlSc2Si2',
                                                'YS_Al', 'YS_Al2', 'YS_Si', 'YS_AlSc2Si2',
                                                'EL_Al', 'EL_Al2', 'EL_Si', 'EL_AlSc2Si2'],
                                         columns=['weight', 'bias'])
        # UTS
        linear_coef_allRE.iloc[0, 0] = fitting_w_1
        linear_coef_allRE.iloc[0, 1] = fitting_b_1
        linear_coef_allRE.iloc[1, 0] = fitting_w_2
        linear_coef_allRE.iloc[1, 1] = fitting_b_2
        linear_coef_allRE.iloc[2, 0] = fitting_w_3
        linear_coef_allRE.iloc[2, 1] = fitting_b_3
        linear_coef_allRE.iloc[3, 0] = fitting_w_4
        linear_coef_allRE.iloc[3, 1] = fitting_b_4
        # YS
        linear_coef_allRE.iloc[4, 0] = fitting_w_5
        linear_coef_allRE.iloc[4, 1] = fitting_b_5
        linear_coef_allRE.iloc[5, 0] = fitting_w_6
        linear_coef_allRE.iloc[5, 1] = fitting_b_6
        linear_coef_allRE.iloc[6, 0] = fitting_w_7
        linear_coef_allRE.iloc[6, 1] = fitting_b_7
        linear_coef_allRE.iloc[7, 0] = fitting_w_8
        linear_coef_allRE.iloc[7, 1] = fitting_b_8
        # EL
        linear_coef_allRE.iloc[8, 0] = fitting_w_9
        linear_coef_allRE.iloc[8, 1] = fitting_b_9
        linear_coef_allRE.iloc[9, 0] = fitting_w_10
        linear_coef_allRE.iloc[9, 1] = fitting_b_10
        linear_coef_allRE.iloc[10, 0] = fitting_w_11
        linear_coef_allRE.iloc[10, 1] = fitting_b_11
        linear_coef_allRE.iloc[11, 0] = fitting_w_12
        linear_coef_allRE.iloc[11, 1] = fitting_b_12

        linear_coef_allRE.to_csv(
            path + 'linear_coef_allRE.csv', float_format='%.2f')
        # plt.show()

    # 获取数据

    x, y_UTS, y_YS, y_EL, EL_Sc = get_training_data(
        training_data_file_path)
    x_predict, EL_Sc_predict = get_predicting_data(
        predicting_data_file_path)

    # 执行正则化，并记住训练集数据的正则化规则,运用于测试集数据

    x_scaler = StandardScaler().fit(x)
    x_standarded = torch.from_numpy(x_scaler.transform(x)).float()
    x_standarded_predict = torch.from_numpy(
        x_scaler.transform(x_predict)).float()
    # print(x_standarded_predict[:, 0].min().numpy())
    # -1.9541031
    # print(x_standarded_predict[:, 0].max().numpy())
    # 0.9373831
    # print(x_standarded_predict[:, 1].min().numpy())
    # -1.2294818
    # print(x_standarded_predict[:, 1].max().numpy())
    # 1.8249916
    # print(x_standarded_predict[:, 2].min().numpy())
    # -1.7462678
    # print(x_standarded_predict[:, 2].max().numpy())
    # 2.0898392
    # print(x_standarded_predict[:, 3].min().numpy())
    # -1.6523643
    # print(x_standarded_predict[:, 3].max().numpy())
    # 1.2637262
    xRe = pd.DataFrame(data=x_standarded_predict.numpy(),
                       columns=['Al', 'Al2', 'Si', 'Al2Si2Sr'])
    xRe.to_csv(path + 'xRe.csv', float_format='%.2f')

    # 调用模型进行预测

    y_list = torch.cat((y_UTS, y_YS, y_EL), 1)
    with torch.no_grad():
        # 预测
        y_predict = predict(model_path, x_standarded_predict)

        real_value = y_list.numpy().flatten()
        predict_value = predict(model_path, x_standarded).numpy().flatten()
        R_value = correlation_coefficient(real_value, predict_value)
        # print(R_value)

        y_scaler = StandardScaler().fit(y_predict)
        y_standarded = torch.from_numpy(y_scaler.transform(y_list)).float()
        y_standarded_predict = torch.from_numpy(
            y_scaler.transform(y_predict)).float()
        # print(y_standarded_predict[:, 0].min().numpy())
        # -3.0705857
        # print(y_standarded_predict[:, 0].max().numpy())
        # 1.7681789
        # print(y_standarded_predict[:, 1].min().numpy())
        # -3.988668
        # print(y_standarded_predict[:, 1].max().numpy())
        # 1.9698732
        # print(y_standarded_predict[:, 2].min().numpy())
        # -2.9861753
        # print(y_standarded_predict[:, 2].max().numpy())
        # 1.2088449

        if np.isnan(y_predict.numpy().any()):
            print('==============Prediction run out of range===============')
        else:
            print('==================Prediction complete===================')

            # 数据可视化(散点图)
            draw_scatter(EL_Sc.numpy(), y_list.numpy(),
                         EL_Sc_predict.numpy(), y_predict.numpy(),
                         error, add, R_value)

            # 综合力学性能计算及可视化
            # data_process(path, EL_Si_predicting.numpy(), EL_Mg_predicting.numpy())

            # 绘制相分数(正则化)-性能(正则化)关系图
            draw_relation_allRE(x_standarded_predict.numpy(),
                                y_standarded_predict.numpy(),
                                'Al')
            draw_relation_allRE(x_standarded_predict.numpy(),
                                y_standarded_predict.numpy(),
                                'Al2')
            draw_relation_allRE(x_standarded_predict.numpy(),
                                y_standarded_predict.numpy(),
                                'Si')
            draw_relation_allRE(x_standarded_predict.numpy(),
                                y_standarded_predict.numpy(),
                                'AlSc2Si2')
            draw_relation_allRE_whole(x_standarded_predict.numpy(),
                                      y_standarded_predict.numpy())


if __name__ == '__main__':
    main(parameters_list)
