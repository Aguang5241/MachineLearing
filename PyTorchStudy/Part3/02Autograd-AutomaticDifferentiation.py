import torch

# Tensor
# Create a tensor and set requires_grad=True to track computation with it
x = torch.ones(2, 2, requires_grad=True)
print(x)
# tensor([[1., 1.],
#         [1., 1.]], requires_grad=True)

# Do a tensor operation
y = x + 2
print(y)
# tensor([[1., 1.],
#         [1., 1.]], requires_grad=True)

# y was created as a result of an operation, so it has a grad_fn
print(y.grad_fn)
# <AddBackward object at 0x0000021F32EC3C18>

# Do more operations on y
z = y * y * 3
out = z.mean()
print(z)
# tensor([[27., 27.],
#         [27., 27.]], grad_fn=<MulBackward>)
print(out)
# tensor(27., grad_fn=<MeanBackward1>)

# .requires_grad_( ... ) changes an existing Tensor’s requires_grad flag in-place. The input flag defaults to False if not given
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
# False
a.requires_grad_(True)
print(a.requires_grad)
# True
b = (a * a).sum()
print(b)
# tensor(7.9161, grad_fn=<SumBackward0>)
print(b.grad_fn)
# <SumBackward0 object at 0x000002065BC03BE0>

# Gradient
# Let’s backprop now. out is a scalar
# out.backward(torch.tensor(1.0))
out.backward()
print(x.grad)
# tensor([[4.5000, 4.5000],
#         [4.5000, 4.5000]])

# Now let’s take a look at an example of vector-Jacobian product
m = torch.ones(3, requires_grad=True)
n = m * 2
while n.data.norm() < 1000:
    n *= 2
print(n)
# tensor([1024., 1024., 1024.], grad_fn=<MulBackward>)

# n is no longer a scalar
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
n.backward(v)
print(m.grad)
# tensor([ 102.4000, 1024.0000,    0.1024])

# You can also stop autograd from tracking history on Tensors with .requires_grad=True either by wrapping the code block in with torch.no_grad()
print(x.requires_grad)
# True
print((x ** 2).requires_grad)
# True
with torch.no_grad():
    print((x ** 2).requires_grad)
    # False

# Or by using .detach() to get a new Tensor with the same content but that does not require gradients
print(x.requires_grad)
# True
y = x.detach()
print(y.requires_grad)
# False
print(x.eq(y).all())
# tensor(1, dtype=torch.uint8)