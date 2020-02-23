import torch
# 1.Tensor
# Construct a 5x3 matrix, uninitialized
x01 = torch.empty(5, 3)
print(x01)
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]])

# Construct a randomly initialized matrix
x02 = torch.rand(5, 3)
print(x02)
# tensor([[0.7168, 0.4827, 0.0129],
#         [0.5516, 0.4749, 0.1798],
#         [0.4293, 0.5737, 0.2047],
#         [0.2033, 0.7312, 0.4394],
#         [0.6203, 0.8661, 0.8887]])

# Construct a matrix filled zeros and of dtype long
x03 = torch.zeros(5, 3, dtype=torch.long)
print(x03)
# tensor([[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]])

# Construct a tensor directly from data
x04 = torch.tensor([5.5, 3])
print(x04)
# tensor([5.5000, 3.0000])

# Create a tensor based on an existing tensor
# new_* methods take in sizes: new_zeros; new_empty
x05 = x04.new_ones(5, 3, dtype=torch.double)
print(x05)
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]], dtype=torch.float64)
# *_like methods: ones_like; zeros_like; empty_like...
x06 = torch.randn_like(x05, dtype=torch.float)
print(x06)
# tensor([[ 0.3248,  0.9863, -0.5039],
#         [ 1.0277,  0.1642, -0.4388],
#         [-1.0968,  0.3399, -1.1493],
#         [ 0.3432,  1.6501, -0.8363],
#         [-0.4128, -0.3045,  1.1941]])

# Get its size
print(x06.size())
# torch.Size([5, 3])
# torch.Size is in fact a tuple, so it supports all tuple operations.

# 2.Operations
# Addition: syntax 1
x07 = torch.tensor([[1, 2], [3, 4]])
x08 = torch.tensor([[1, 2], [3, 4]])
print(x07 + x08)

# Addition: syntax 2
print(torch.add(x07, x08))

# Addition: providing an output tensor as argument
result01 = torch.empty(x07.size(), dtype=torch.int64)
torch.add(x07, x08, out=result01)
print(result01)

# Addition: in-place
x08.add_(x07)
print(x08)
# tensor([[2, 4],
#         [6, 8]])

# Any operation that mutates a tensor in-place is post-fixed with an _, and will change x
x09 = torch.tensor([1, 2])
x10 = torch.tensor([3, 4])
x10.copy_(x09)
print(x10)
# tensor([1, 2])

# Resizing
x11 = torch.ones(4 ,4)
x12 = x11.view(16)
# the size -1 is inferred from other dimensions
x13 = x12.view(-1, 8)
print(x11, '\n', x12, '\n', x13)
# tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.]])
#  tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
#  tensor([[1., 1., 1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1., 1., 1.]])

# If you have a one element tensor, use .item() to get the value as a Python number
x14 = torch.rand(1)
print(x14)
# tensor([0.6395])
print(x14.item())
# 0.6394924521446228

# Numpy bridge
# Converting a Torch Tensor to a NumPy Array
x15 = torch.ones(5)
print(x15)
# tensor([1., 1., 1., 1., 1.])
x16 = x15.numpy()
print(x16)
# [1. 1. 1. 1. 1.]

# See how the numpy array changed in value.
x15.add_(1)
print(x15)
# tensor([2., 2., 2., 2., 2.])
print(x16)
# [2. 2. 2. 2. 2.]

# Converting NumPy Array to Torch Tensor
import numpy as np
x17 = np.ones(5)
x18 = torch.from_numpy(x17)
np.add(x17, 1, out=x17)
print(x17)
# [2. 2. 2. 2. 2.]
print(x18)
# tensor([2., 2., 2., 2., 2.], dtype=torch.float64)