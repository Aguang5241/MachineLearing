import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 创建分类
X, Y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, scale=100, random_state=22)
# 可视化
# plt.scatter(X[:, 0], X[:, 1], c=Y)
# plt.show()
# 归一化
X = preprocessing.scale(X)
# 划分训练集和测试集
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
# 定义模型
clf = SVC()
# 拟合
clf.fit(trainX, trainY)
# 测试拟合效果
print(clf.score(testX, testY))