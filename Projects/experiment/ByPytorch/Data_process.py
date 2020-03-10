import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler


# UTS + alog10(EL)
def process_1(path, a):
    data = pd.read_csv(path + 'model-v1.4.3.csv')
    data_calculated = data.values[:, 0] + a * np.log10(data.values[:, 2])
    return data, data_calculated


# YS + alog10
def process_2(path, a):
    data = pd.read_csv(path + 'model-v1.4.3.csv')
    data_calculated = data.values[:, 1] + a * np.log10(data.values[:, 2])
    return data, data_calculated


# aUTS + bYS + cEL
def process_3(path, a, b, c):
    data = pd.read_csv(path + 'model-v1.4.3.csv')
    mms = MinMaxScaler()
    data_processed = mms.fit_transform(data.values)
    data_calculated = a * data_processed[:, 0] + b * \
        data_processed[:, 1] + c * data_processed[:, 2]
    return data, data_calculated


def data_process(path, x, y, data, data_calculated, a, b, c):
    # 获取最值索引
    max_index = data_calculated.tolist().index(max(data_calculated))
    print('========================Results=========================')
    results = 'Si: ' + str(x[max_index][0]) + \
        '\nMg: ' + str(y[max_index][0]) + \
        '\nUTS: ' + str(data.values[max_index, 0]) + \
        '\nYS: ' + str(data.values[max_index, 1]) + \
        '\nEL: ' + str(data.values[max_index, 2])
    print(results)
    f = open(path + 'results.csv', 'w')
    f.write(results)
    f.close()
    # 可视化
    sns.set(font="Times New Roman")
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('The coefficient a is: %.3f; ' % a +
                 'The coefficient b is: %.3f; ' % b +
                 'The coefficient c is: %.3f' % c)
    ax.set_xlabel('Si')
    ax.set_ylabel('Mg')
    ax.set_zlabel('General Performance')
    ax.scatter(x, y, data_calculated)
    ax.scatter(x[max_index], y[max_index],
               data_calculated[max_index], color='red', s=50)
    plt.savefig(path + 'aUTS_bYS_cEL/model-v1.4.3(a%.2fb%.2fc%.2f).png' % (a, b, c))
    # plt.show()


def getTestingData(file_path):
    data = pd.read_csv(file_path)
    x = torch.from_numpy(data.loc[:, 'PH_Al':'PH_AlSc2Si2'].values).float()
    EL_Si = torch.unsqueeze(
        (torch.from_numpy(data['EL_Si'].values)), dim=1).float()
    EL_Mg = torch.unsqueeze(
        (torch.from_numpy(data['EL_Mg'].values)), dim=1).float()
    return x, EL_Si, EL_Mg


def main():
    path = 'Projects/Experiment/res/model-v1.4.3/Part1/0.866(ok)/'
    testing_data_file_path = 'Projects/Experiment/res/TestingDataFiltered.csv'
    x_testing, EL_Si_test, EL_Mg_test = getTestingData(testing_data_file_path)
    a_list = np.linspace(0, 1, 5)
    b_list = np.linspace(0, 1, 5)
    c_list = np.linspace(0, 1, 5)
    for a in a_list:
        # data, data_calculated = process_1(path, a)
        # data, data_calculated = process_2(path, a)
        for b in b_list:
            for c in c_list:
                data, data_calculated = process_3(path, a, b, c)
                data_process(path, EL_Si_test.numpy(),
                             EL_Mg_test.numpy(), data, data_calculated, a, b, c)


if __name__ == '__main__':
    main()
