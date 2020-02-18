import torch

# x = torch.rand(3, requires_grad=True)
x = torch.tensor([2.0], requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
  y = y * 2
print(y)
# tensor([ 470.1743,  648.9385, 1718.2977], grad_fn=<MulBackward>)
# gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
gradients = torch.tensor([0.1], dtype=torch.float)
y.backward(gradients)
print(x.grad)
# tensor([ 204.8000, 2048.0000, 0.2048])