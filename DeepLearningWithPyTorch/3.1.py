from torch.nn import Linear
import torch

def main():
    my_linear = Linear(in_features=10, out_features=5, bias=True)
    inp = torch.randn(1, 10, requires_grad=True)
    print(inp)
    # tensor([[ 0.6753, -0.1242, -0.2882, -1.2005, -1.1807, -0.2852, -1.5381, -1.0471, -1.4856,  0.2185]], requires_grad=True)
    print(my_linear(inp))
    # tensor([[ 0.8444,  0.6141, -1.1068, -0.0438,  1.0435]], grad_fn=<AddmmBackward>)
    print(my_linear.weight)
    # Parameter containing:
    # tensor([[-0.2872,  0.2844, -0.1930, -0.2874,  0.0137, -0.1668, -0.0744,  0.1109,
    #         -0.2629,  0.1396],
    #         [-0.3080,  0.0206,  0.2366,  0.2033, -0.0213,  0.2471, -0.1887, -0.2771,
    #         -0.2938,  0.2766],
    #         [-0.2833, -0.0285, -0.0911,  0.2500, -0.0843, -0.2197,  0.1723,  0.0697,
    #         0.1081, -0.0940],
    #         [ 0.2563,  0.0753,  0.0822, -0.0085, -0.1673,  0.2199,  0.2523, -0.1610,
    #         0.0424,  0.3131],
    #         [ 0.1232,  0.1502, -0.0503,  0.2908, -0.2772, -0.2394, -0.2930, -0.3077,
    #         -0.1837,  0.3083]], requires_grad=True)
    print(my_linear.bias)
    # Parameter containing:
    # tensor([ 0.2221,  0.1050, -0.2881, -0.1148, -0.1952], requires_grad=True)

if __name__ == '__main__':
    main()