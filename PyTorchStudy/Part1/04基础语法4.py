import torch

if torch.cuda.is_available():
    device = torch.device("cuda")          # torch.device 将张量移动到指定的设备中
    y = torch.ones_like(x, device=device)  # 直接从 GPU 创建张量
    x = x.to(device)                       # 或者直接使用 .to("cuda") 将张量移动到 cuda 中
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # .to 也会对变量的类型做更改
else:
  print('no CUDA')