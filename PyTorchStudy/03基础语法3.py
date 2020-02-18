import torch
import numpy as np

# 张量转换为numpy
a = torch.ones(5)
print(a)
# tensor([1., 1., 1., 1., 1.])
b = a.numpy()
print(b)
# [1. 1. 1. 1. 1.]
a.add_(1)
print(a)
# tensor([2., 2., 2., 2., 2.])
print(b)
# [2. 2. 2. 2. 2.]
# numpy 转换为张量
x = np.ones(5)
y = torch.from_numpy(x)
print(x)
# [1. 1. 1. 1. 1.]
print(y)
# tensor([1., 1., 1., 1., 1.], dtype=torch.float64)
np.add(x, 1, out=x)
print(x)
# [2. 2. 2. 2. 2.]
print(y)
# tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
