import torch
from torch import nn


# #############不含模型参数的自定义层################

class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()

# layer = CenteredLayer()
# print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)))

# # 构造一个更复杂的模型
# net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
# y = net(torch.rand(4, 8))
# print(y.mean().item())


# #############含模型参数的自定义层################
class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))
    
    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params)
        return x

net = MyDense()
print(net)

class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(4, 4)),
            'linear2': nn.Parameter(torch.randn(4, 1))
        })
        # 新增
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))})
    
    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])
    
net = MyDictDense()
print(net)