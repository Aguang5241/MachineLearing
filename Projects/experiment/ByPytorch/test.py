import torch
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

X_train = np.array([[1., 1.,  2.],
                    [2.,  0.,  0.],
                    [0.,  1., 1.]])

mms = MinMaxScaler()
data = mms.fit_transform(X_train)
print(data)
print(mms.scale_)