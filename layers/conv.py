import math
import unittest

import torch
import torch.nn as nn


class Conv2DStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers, stride, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define a stack of 2D convolution layers
        self.conv_layers = nn.Sequential()
        self.out_channels = out_channels
        for l in range(num_layers):
            self.conv_layers.add_module(f'conv_{l}', nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, stride)))
            self.conv_layers.add_module(f'relu_{l}', nn.ReLU())
            in_channels = out_channels  # Update the input channel for the next layer

    def forward(self, x):
        # Input x should be a BxNxT tensor
        B, N, T = x.size()

        # Reshape the input tensor to B * N x 1 x T for 2D convolution
        x = x.reshape(B*N, 1, 1, T)

        # Apply the stack of convolution layers
        x = self.conv_layers(x)

        # Reshape the output tensor to BxN
        x = x.view(B, N, self.out_channels, -1).transpose(1, 2)

        return x


def top_k_graph(scores, h, k, dim):
    values, idx = torch.topk(scores, k, dim)
    new_h = torch.gather(h, dim=dim, index=idx.repeat(1, 1, h.shape[2]))
    new_h = torch.mul(new_h, values)
    return new_h, idx


class Pool(nn.Module):
    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h, scores):
        Z = self.drop(scores)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        return top_k_graph(scores, h, self.k, 1)


class SparseBasisSelector(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers, stride, k, in_dim, p):
        super(SparseBasisSelector, self).__init__()
        self.conv_stack = Conv2DStack(in_channels, out_channels, kernel_size, num_layers, stride)
        self.pool = Pool(k, in_dim, p)

    def forward(self, h):
        conv_out = self.conv_stack(h)
        conv_out = conv_out.transpose(1, 2).flatten(start_dim=2)
        h_new, idx = self.pool(h, conv_out)
        return h_new, idx


class TestConv2DStack(unittest.TestCase):
    def test_forward(self):
        B, N = 2, 50  # Batch size, number of variables, number of time points
        in_channels = 1
        out_channels = 64
        kernel_size = 5
        num_layers = 3
        stride = 3

        for T in range(65, 100):
            # Create a Conv2DStack instance
            conv_stack = Conv2DStack(in_channels, out_channels, kernel_size, num_layers, stride)

            # Generate random input data
            input_tensor = torch.randn(B, N, T)

            # Forward pass
            output_tensor = conv_stack(input_tensor)

            # Check if the output tensor has the correct shape
            expected_width = T
            for i in range(num_layers):
                expected_width = (expected_width - kernel_size + stride) // stride
            expected_shape = (B, out_channels, N, expected_width)
            self.assertEqual(expected_shape, output_tensor.size())


class TestPool(unittest.TestCase):
    def test_forward(self):
        k, in_dim, p = 10, 100, 0.1
        pool = Pool(k, in_dim, p,)
        h = torch.randn((2, 20, in_dim))
        h_new, idx = pool(h)
        self.assertEqual((2, k, in_dim), h_new.size())


class TestSparseBasisSelector(unittest.TestCase):
    def test_forward(self):
        B, N = 2, 50  # Batch size, number of variables, number of time points
        in_channels = 1
        out_channels = 512
        kernel_size = 5
        stride = 2
        k, p = 10, 0.1
        T = 32
        num_layers = 3
        in_dim = T
        for i in range(num_layers):
            in_dim = (in_dim - kernel_size + stride) // stride
        in_dim *= out_channels
        sbs = SparseBasisSelector(in_channels, out_channels, kernel_size, num_layers, stride, k, in_dim, p)
        input_tensor = torch.randn(B, N, T)
        output_tensor, idx = sbs(input_tensor)
        self.assertEqual((input_tensor.shape[0], k, input_tensor.shape[2]), output_tensor.shape)
