import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def main():
    # 处理三项性能，得到综合性能（如何定义最佳力学性能有待斟酌）
    # def getComprehensivePerformance(UST, YS, EL):
    #     return comprehensivePerformance

    # 寻找相分数（四个）与指定性能之间的关系（核心）
    def model(phaseAl, phaseSi, phaseAlSc2Si2, phaseMg2Si, performance):
        return

    # 绘制 3D 图像
    def draw3DFig(elementSi, elementMg, performance):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(elementSi, elementMg, performance)
        plt.show()

    # 获取指定数据
    def getData(dataName):
        filename = 'Projects/rawData.csv'
        rawData = pd.read_csv(filename)
        return data

    # overallView
    # sns.pairplot(rawData, hue='Alloy')


if __name__ == '__main__':
    main()
