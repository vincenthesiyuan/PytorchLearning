import torch
import numpy as np

def tensor_op():
    """
    关于tensor的一些操作
    """

    # 加法操作
    x = torch.ones(5, 3)
    y = torch.rand(5, 3)
    print(x + y)

    print(torch.add(x, y))

    result = torch.empty(5, 3)
    torch.add(x, y, out = result)
    print("result:", result)  # 指定输出

    y.add_(x)  # inplace   
    print("inplace y:", y)

    # 索引 索引出来的结果与原数据共享内存
    y = x[0, :]
    y += 1  # 原tensor x也被修改
    print(y)
    print(x[0, :])  

    # 用view()来改变tensor的形状
    y = x.view(15)
    z = x.view(-1, 5)
    print(x.size(), y.size(), z.size())




tensor_op()

def boardcasting_pra():
    """
    2.2.3 广播机制
    """
    x = torch.arange(1, 3).view(1, 2)
    print(x)

    y = torch.arange(1, 4).view(3, 1)
    print(y)

    print(x + y)

def chapter2_2_4():
    """
    2.2.4 运算的内存开销
    """
    x = torch.tensor([1, 2])
    y = torch.tensor([3, 4])
    id_before = id(y)

    # y = y + x  # 会新开内存
    # print(id(y) == id_before)

    #torch.add(x, y, out = y)
    y += x  #True
    print(id(y) == id_before)

# chapter2_2_4()

def chapter2_2_5():
    """
    2.2.5 tensor和numpy互相转换

    numpy() 和 from_numpy()将tensor和numpy中的数组相互转换，其是共享内存
    而将numpy中的array用torch.tensor()转换，其是进行数据拷贝，返回的tensor和原来的书记不再共享内存
    """
    a = torch.ones(5)
    b = a.numpy()
    print(a, b)

    a += 1
    print(a, b)

    b += 1
    print(a, b)

    # torch.tensor()是进行数据拷贝，返回的Tensor和原来的数据不再共享内存
    d = np.ones(5)
    c = torch.tensor(d)
    d += 1
    print(c, d)

# chapter2_2_5()

def  chapter2_2_6():
    """
    用方法to()可以将Tensor在CPU和GPU（需要硬件支持）之间相互移动。
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.tensor([1, 2])
        y = torch.ones_like(x, device = device)
        x = x.to(device)
        z = y + x
        print(z)
        print(z.to("cpu", torch.double))
    
# chapter2_2_6()