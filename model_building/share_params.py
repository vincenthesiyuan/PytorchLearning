"""
如何访问和初始化模型参数，以及如何在多个层之间共享同一份模型参数
"""

import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()

# print(type(net.named_parameters()))
# for name, param in net.named_parameters():
#     print(name, param.size())

# 对于Sequential类构造的神经网络，可以通过[]来访问网络的任一层。
for name, param in net[0].named_parameters():
    print(name, param.size())

