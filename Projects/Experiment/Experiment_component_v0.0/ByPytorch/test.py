import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# loss = torch.tensor([[-3, 3, 1],
#                      [-3, 3, 1],
#                      [-3, 3, 1],
#                      [-3, 3, 1],
#                      [-3, 3, 1],
#                      [-3, 3, 1]]).float()

# error = torch.tensor([[3, 3, 1],
#                       [3, 3, 1],
#                       [3, 3, 1],
#                       [3, 3, 1],
#                       [3, 3, 1],
#                       [3, 3, 1]]).float()
# print(loss)
# print(error)

# # loss_func = torch.nn.MSELoss()
# loss_func = torch.nn.CosineEmbeddingLoss()
# loss_ = loss_func(loss, error, torch.tensor([[1], [1], [1]]).float())
# print(loss_)

############################################################################

# X_train = np.array([[1., 1.,  2.],
#                     [2.,  0.,  0.],
#                     [0.,  1., 1.]])

# mms = MinMaxScaler()
# data = mms.fit_transform(X_train)
# print(data)
# print(mms.scale_)

############################################################################

# x = torch.Tensor([1, 1, 0.1])
# y= x.repeat(4, 1)
# print(y)

############################################################################

data = pd.read_csv(r'Projects/Experiment/res/model-v1.2.6/Part3(8x3)/0.900(5.5-0.45-1)/model-v1.2.6.csv')
print(data.values[250, :])