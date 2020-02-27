import torch

# x1 = torch.tensor([[1, 2, 3, 4],
#                    [1, 2, 3, 4],
#                    [1, 2, 3, 4]]).float()
# x2 = torch.tensor([[1, 2, 3, 4],
#                    [1, 2, 3, 4],
#                    [1, 2, 3, 4]]).float()
# x1 = torch.autograd.Variable(torch.tensor([[1], [2], [3]]).float())
# x2 = torch.autograd.Variable(torch.tensor([[1], [2], [0]]).float())
x1 = torch.autograd.Variable(torch.tensor([[1, 1, 1],
                                           [2, 2, 2],
                                           [3, 3, 3]]).float())
x2 = torch.autograd.Variable(torch.tensor([[1, 1, 1],
                                           [2, 2, 2],
                                           [0, 0, 0]]).float())

y = torch.tensor([-1, -1, -1]).float()
# y = torch.empty(3)
# y = torch.empty(3).bernoulli_()
# y = torch.empty(3).bernoulli_().mul_(2)
# y = torch.empty(3).bernoulli_().mul_(2).sub_(1)
l = torch.nn.CosineEmbeddingLoss()

print(y)


def ang(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))


# By definition: the loss value should be mean of (1 - ang(x1[0], x2[0])), ang(x1[1], x2[1]) and 0
# Which is approx 0.326
print(l(x1, x2, y).item())  # prints 0.325969
