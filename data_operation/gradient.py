import torch


def tensor_op():
    x = torch.ones(2, 2, requires_grad=True)
    print(x)
    print(x.grad_fn)

    y = x + 2
    print(y)
    print(y.grad_fn)

    z = y * y * 3
    out = z.mean()
    print(z, out)

def inplace_req_grad():
    """
    用inplace的方式来改变requires_grad
    """
    a = torch.randn(2, 2)  # 默认requires_grad = false
    a = ((a * 3) / (a - 1))
    print(a.requires_grad)  # false
    a.requires_grad_(True)
    print(a.requires_grad)  # true
    b = (a * a).sum()
    print(b.grad_fn)


def grad_accumulation():
    """
    grad在反向传播过程中是累加的，这意味着每一次运行反向传播，梯度都会累加之前的梯度。所以一般在反向传播之前把梯度清零。
    """
    x = torch.ones(2, 2, requires_grad=True)
    print(x)
    print(x.grad_fn)

    y = x + 2
    print(y)
    print(y.grad_fn)

    z = y * y * 3
    out = z.mean()
    print(z, out)

    out.backward()
    print(x.grad)

    # 再来反向传播一次，注意grad是累加的
    out2 = x.sum()
    out2.backward()
    print(x.grad)

    out3 = x.sum()
    x.grad.data.zero_()
    out3.backward()  
    print(x.grad)


def interrupt_grad():
    x = torch.tensor(1.0, requires_grad=True)
    y1 = x ** 2 
    with torch.no_grad():
        y2 = x ** 3
    y3 = y1 + y2

    print(x.requires_grad)
    print(y1, y1.requires_grad) # True
    print(y2, y2.requires_grad) # False
    print(y3, y3.requires_grad) # True

    y3.backward()
    print(x.grad)

#interrupt_grad()

def change_num():
    """
    如果我们想要修改tensor的数值，但是又不希望被autograd记录（即不会影响反向传播），那么我么可以对tensor.data进行操作。
    """
    x = torch.ones(1,requires_grad=True)

    print(x.data) # 还是一个tensor
    print(x.data.requires_grad) # 但是已经是独立于计算图之外

    y = 2 * x
    x.data *= 100 # 只改变了值，不会记录在计算图，所以不会影响梯度传播

    y.backward()
    print(x) # 更改data的值也会影响tensor的值
    print(x.grad)

change_num()