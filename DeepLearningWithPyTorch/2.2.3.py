import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)
# tensor([[1., 1.],
#         [1., 1.]], requires_grad=True)
y = x.mean()
print(y)
# tensor(1., grad_fn=<MeanBackward0>)
y.backward()
print(x.grad)
# tensor([[0.2500, 0.2500],
#         [0.2500, 0.2500]])
print(x.grad_fn)
# None
print(x.data)
# tensor([[1., 1.],
#         [1., 1.]])
print(y.grad_fn)
# <MeanBackward0 object at 0x000001BE4C2F2CF8>