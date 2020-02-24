import torch

x = torch.unsqueeze(torch.linspace(-1, 1, 10), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

print(x)
print(y)