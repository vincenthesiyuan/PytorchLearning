"""
    使用autograd的profiler对gpu和cpu的每个操作花销做监控
"""

import torch

x = torch.randn(1, 1).requires_grad_(True)
# print(x)

with torch.autograd.profiler.profile() as prof:
    y = x ** 2
    y.backward()

print(prof)
