# This example uses the only the first feature of the diabetes dataset, in order to illustrate a two-dimensional plot of this regression technique.
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# 载入糖尿病数据
diabetesX, diabetesY = datasets.load_diabetes(return_X_y=True)
# 截取其中的一列属性
diabetesX = diabetesX[:, np.newaxis, 2]
# 分割训练集和测试集
diabetesXTrain = diabetesX[:-20]
diabetesXTest = diabetesX[-20:]
diabetesYTrain = diabetesY[:-20]
diabetesYTest = diabetesY[-20:]
# 定义模型
reg = linear_model.LinearRegression()
# 训练模型
reg.fit(diabetesXTrain, diabetesYTrain)
# 预测
prediction = reg.predict(diabetesXTest)
# 绘图
sns.set()
plt.scatter(diabetesXTest, diabetesYTest, c='red')
plt.plot(diabetesXTest, prediction)
plt.show()
# 输出结果
print('The coefficients:\n', reg.coef_) # 系数
print('The mean square error: %.2f\n',
      mean_squared_error(diabetesYTest, prediction)) # 均方误差
print('The coefficient of determination: %.2f\n',
      r2_score(diabetesYTest, prediction)) # 确定系数
