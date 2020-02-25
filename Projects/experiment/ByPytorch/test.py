import torch
import numpy as np
from sklearn import preprocessing

X = np.array([[1., -1.,  2.],
              [2.,  0.,  0.],
              [0.,  1., -1.]])

scaler = preprocessing.StandardScaler().fit(X)

print(scaler.mean_)


print(scaler.transform(X))


# 可以直接使用训练集对测试集数据进行转换
print(scaler.transform([[-1.,  1., 0.]]))

print(scaler.mean_)
