import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        # 初始化父类，即 nn.Module
        super(Net, self).__init__()
        # 1 input; 6 output; 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # y = wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
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

params = list(net.parameters())
print(len(params))
# 10
print(params[0].size())  # conv1's .weight
# torch.Size([6, 1, 3, 3])

input = torch.randn(1, 1, 32, 32)
print(input)
# tensor([[[[ 0.2673,  0.4926, -0.2224,  ...,  1.1279, -0.0555,  0.8644],
#           [-0.1330, -0.1882, -0.0153,  ..., -0.6634,  0.6310,  0.9217],
#           [-0.9258,  0.2748,  0.8340,  ...,  0.7596,  0.6761, -0.5563],
#           ...,
#           [ 0.1125, -0.9503, -0.1606,  ...,  0.9612,  0.7760,  0.3352],
#           [ 0.0691, -0.5333,  2.8262,  ...,  2.9444,  0.3148, -0.6768],
#           [ 0.7549,  0.6300,  0.6570,  ...,  2.0347, -0.0806, -1.5532]]]])
out = net(input)
print(out)
# tensor([[-0.1045,  0.0563,  0.0999, -0.0701, -0.0302,  0.0066,  0.0010,  0.0546,
#           0.0968,  0.0938]], grad_fn=<ThAddmmBackward>)