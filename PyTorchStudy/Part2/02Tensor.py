import torch

device = torch.device('cpu')
# device = torch.device('cuda')

# N is batch size; D_in is input dimension
# H is hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random data
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# Random initialize weight
w1 = torch.randn(D_in, H, device=device)
w2 = torch.randn(H, D_out, device=device)

lr = 1e-6
for t in range(500):
    # forward pass: compute the predict y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # backprop to compute gradient of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # updateweight
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
