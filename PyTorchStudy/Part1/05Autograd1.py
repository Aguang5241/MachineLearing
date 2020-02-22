import torch

# requires_grad=True 用来追踪计算历史
x = torch.ones(2, 2, requires_grad=True)
print(x)
# tensor([[1., 1.],
#         [1., 1.]], requires_grad=True)
# 对 x 进行操作
y = x + 2
print(y)
# tensor([[3., 3.],
#         [3., 3.]], grad_fn=<AddBackward>)
print(y.grad_fn)
# <AddBackward object at 0x000002CF0C91FC88
# 对 y 进行操作
z = y * y * 3
print(z)
# tensor([[27., 27.],
#         [27., 27.]], grad_fn=<MulBackward>)
print(z.grad_fn)
# <MulBackward object at 0x000001F2FFE6FC88>
out = z.mean()
print(out)
# tensor(27., grad_fn=<MeanBackward1>)

# 反向追踪
out.backward()
print(x.grad)
# tensor([[4.5000, 4.5000],
#         [4.5000, 4.5000]])