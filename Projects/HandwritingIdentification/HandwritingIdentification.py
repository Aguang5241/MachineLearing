from sklearn import svm, datasets
import matplotlib.pyplot as plt

def main():
    # 采用 svc 技术
    svc = svm.SVC(C=100, gamma=0.001)
    # 导入手写字体图像数据
    digits = datasets.load_digits()
    # 打印概述信息
    # print(digits.DESCR)
    # 展示
    # plt.imshow(digits.images[2], cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.show()
    # 获取数据集数量
    # print(digits.data.size)
    # 115008 (1797 * 64)
    # print(digits.target.size)
    # 1797

    # 拟采用1791个数据做训练，6个数据做验证
    # 以下为其中的 6 个待测试图像
    # plt.subplot(321)
    # plt.imshow(digits.images[1791], cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.subplot(322)
    # plt.imshow(digits.images[1792], cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.subplot(323)
    # plt.imshow(digits.images[1793], cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.subplot(324)
    # plt.imshow(digits.images[1794], cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.subplot(325)
    # plt.imshow(digits.images[1795], cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.subplot(326)
    # plt.imshow(digits.images[1796], cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.show()

    # 使用 svc 估计器进行学习
    svc.fit(digits.data[1:1790], digits.target[1:1790])
    svc.predict(digits.target[1791:1797])
    # array=[4. 9. 0. 8. 9. 8.]

if __name__ == '__main__':
    main()
