import torch
from torch import nn
from torch.nn import init

"""
    模型参数的访问、初始化和共享
"""

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()

# 访问模型参数
# print(type(net.named_parameters()))
# for name, param in net.named_parameters():
#     print(name, param.size)

# for name, param in net[0].named_parameters():
#     print(name, param.size(), type(param))


# 初始化模型参数
# for name, param in net.named_parameters():
#     if 'weight' in name:
#         init.normal_(param, mean=0, std=0.01)
#         print(name, param.data)


"""
    令权重有一半概率初始化为0
    另一半概率初始化为[−10,−5][−10,−5]和[5,10][5,10]两个区间里均匀分布的随机数
"""
def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)

print("\n")
####################################################


# 共享模型参数
def share_params():
    linear = nn.Linear(1, 1, bias=False)
    net = nn.Sequential(linear, linear)
    print(net)

    for name, param in net.named_parameters():
        init.constant_(param, val=3)
        print(name, param.data)
    
    print(id(net[0]) == id(net[1]))
    print(id(net[0].weight) == id(net[1].weight))

share_params()

