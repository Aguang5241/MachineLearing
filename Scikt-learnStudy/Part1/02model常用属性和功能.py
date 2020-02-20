from sklearn.linear_model import LinearRegression
from sklearn import datasets


loadedData = datasets.load_boston()
dataX = loadedData.data
dataY = loadedData.target
# 建立模型
model = LinearRegression()
# 训练模型
model.fit(dataX, dataY)
# 预测数据
print(model.predict(dataX[:4, :]))
# 获取系数
print(model.coef_)
# 获取截距
print(model.intercept_)
# 模型评估
print(model.score(dataX, dataY))