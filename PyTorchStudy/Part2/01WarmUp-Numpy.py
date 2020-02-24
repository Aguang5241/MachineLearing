import numpy as np

# N is batch size; D_in is input dimension
# H is hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random data
x = np.random.randn(N, D_in)    # 64x1000
y = np.random.randn(N, D_out)   # 64x10

# Random initialize weight
w1 = np.random.randn(D_in, H)   # 1000x100
w2 = np.random.randn(H, D_out)  # 100x10

lr = 1e-6
for t in range(500):
    # forward pass: compute the predict y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # backprop to compute gradient of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # updateweight
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
