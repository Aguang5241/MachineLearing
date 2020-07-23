# 编程实现由给定训练集求解S和G

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def draw(data_list, s_list, g_list):
    sns.set(style="ticks", font="Arial", font_scale=1.5)
    fig = plt.figure(figsize=(8, 6))
    # data
    plt.scatter(data_list[0], data_list[1], s=100)
    plt.scatter(data_list[2], data_list[3], s=100)
    # S-square
    plt.vlines(s_list[0], s_list[2], s_list[3], linestyles='dashed')
    plt.vlines(s_list[1], s_list[2], s_list[3], linestyles='dashed')
    plt.hlines(s_list[2], s_list[0], s_list[1], linestyles='dashed')
    plt.hlines(s_list[3], s_list[0], s_list[1], linestyles='dashed')
    # G-square
    plt.vlines(g_list[0], g_list[2], g_list[3], colors='r', linestyles='dashed')
    plt.vlines(g_list[1], g_list[2], g_list[3], colors='r', linestyles='dashed')
    plt.hlines(g_list[2], g_list[0], g_list[1], colors='r', linestyles='dashed')
    plt.hlines(g_list[3], g_list[0], g_list[1], colors='r', linestyles='dashed')
    plt.show()
    plt.savefig(r'D:\Study\Coding\MachineLearing\IntroductionToMachineLearning\2-4-1.png')


def main():
    data_path = r'D:\Study\Coding\MachineLearing\IntroductionToMachineLearning\2-4.csv'
    data = pd.read_csv(data_path)

    x1 = data.iloc[0:10, 0]
    y1 = data.iloc[0:10, 1]
    x2 = data.iloc[10:15, 0]
    y2 = data.iloc[10:15, 1]
    data_list = [x1, y1, x2, y2]

    g_min_x = x1.min()
    g_max_x = x1.max()
    g_min_y = y1.min()
    g_max_y = y1.max()
    g_list = [g_min_x, g_max_x, g_min_y, g_max_y]

    s_min_x = x2.min()
    s_max_x = x2.max()
    s_min_y = y2.min()
    s_max_y = y2.max()
    s_list = [s_min_x, s_max_x, s_min_y, s_max_y]

    draw(data_list, s_list, g_list)


if __name__ == '__main__':
    main()
