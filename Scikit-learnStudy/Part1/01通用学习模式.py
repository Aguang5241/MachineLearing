import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 载入数据集
iris = datasets.load_iris()
# 获取数据集中的数据内容
irisX = iris.data
# 获取数据集中的种类内容
irisY = iris.target
# print(iris)
# print(irisX)
# print(irisY)

# 分割数据集
trainX, testX, trainY, testY = train_test_split(irisX, irisY, test_size=0.3)
# print(trainY)
# 定义训练模型
knn = KNeighborsClassifier()
# 进行训练
knn.fit(trainX, trainY)
# 进行预测
predictResults = knn.predict(testX)
# 对比预测结果
print('the real results:\n', testY)
print('the prediction:\n', predictResults)