import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

"""
loss function使用 square loss(平方损失)，即训练数据集中所有样本误差的平均来衡量模型预测质量
在模型训练中，希望找出一组模型参数，来使训练样本平均损失最小。

优化算法：
    analytical solution and numerical solution

    大多数深度学习模型并没有解析解，
    在数值解的优化算法中，mini-batch stochastic gradient descent 被广泛使用。

    hyperparameter:  learning rate, batch size

"""


# 数据生成part
num_input = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_input, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
# print(features[0], labels[0])

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)  # index_select(dim, index)

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break




