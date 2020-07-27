import torch
import numpy as np

# 加法
def add_tensors(a, b):
    c = a + b
    print(c)
    d = torch.add(a, b)
    print(d)
    a_ = a.add_(5)
    print(a_)

# 乘法
def mul_tensors(a, b):
    c = a * b
    print(c)
    d = a.mul(b)
    print(d)
    a_ = a.mul_(b)
    print(a_)

def main():
    a = torch.tensor([[1, 2],
                    [3, 4]])
    b = torch.tensor([[4, 3],
                    [2, 1]])
    print(a)
    print(b)

    # add_tensors(a, b)

    mul_tensors(a, b)

if __name__ == '__main__':
    main()


