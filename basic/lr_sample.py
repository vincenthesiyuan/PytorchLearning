"""
线性回归的简单实现
"""

import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

# 打印数据集第一行
# for X, y in data_iter:
#     print(X, y)
#     break


# define lr model
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net)

"""
# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

print(net)
print(net[0])
"""

for param in net.parameters():
    print(param)

# initial model parameters
# net写成ModuleList或者Sequential的形式，net.linear就可以用net[0]的形式访问
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.03)
# print(optimizer)
"""
可以对不同的子网络设置不同的学习率，finetune常用
optimizer =optim.SGD([
                # 如果对某个参数不指定学习率，就使用最外层的默认学习率
                {'params': net.subnet1.parameters()}, # lr=0.03
                {'params': net.subnet2.parameters(), 'lr': 0.01}
            ], lr=0.03)

"""

# 调整学习率
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1

# training
num_epochs = 3
for epoch in range(1, num_epochs):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))


dense = net.linear
print(true_w, "\n", dense.weight)
print(true_b, "\n", dense.bias)
