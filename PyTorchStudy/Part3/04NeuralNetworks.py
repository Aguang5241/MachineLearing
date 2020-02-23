import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel; 6 output channel; 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # if the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_feature(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_feature(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
# Net(
#   (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
#   (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
#   (fc1): Linear(in_features=576, out_features=120, bias=True)
#   (fc2): Linear(in_features=120, out_features=84, bias=True)
#   (fc3): Linear(in_features=84, out_features=10, bias=True)
# )

# The learnable parameters of a model are returned by net.parameters()
params = list(net.parameters())
print(len(params))
# 10
print(params[0].size())
# torch.Size([6, 1, 3, 3])

# Loss function
input_value = torch.randn(1, 1, 32, 32)
output = net(input_value)
# a dummy target, for example
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)
# tensor(2.6445, grad_fn=<MseLossBackward>)

# Backprop
# zeroes the gradient buffers of all parameters
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
# conv1.bias.grad before backward
# None
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
# conv1.bias.grad after backward
# tensor([ 0.0097, -0.0020,  0.0082,  0.0077,  0.0030,  0.0058])

# Update the weights
# Stochastic Gradient Descent (SGD): weight = weight - learning_rate * gradient
import torch.optim as optim
lr = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * lr)
# Create optimizer
optimizer = optim.SGD(net.parameters(), lr=lr)
# in training loop
# zero the gradient buffers
optimizer.zero_grad()
loss.backward()
# Does the update
optimizer.step() 