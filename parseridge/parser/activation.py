import math

import torch
from torch import nn


class GELU(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        )


ACTIVATION_FUNCTIONS = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "hard_tanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "elu": nn.ELU,
    "gelu": GELU,
}
