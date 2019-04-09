import math

import torch
from torch import nn


def init_weights_xavier_(network, activation="tanh"):
    """
    Initializes the layers of a given network with random values.
    Bias layers will be filled with zeros.

    Parameters
    ----------
    network : torch.nn object
        The network to initialize.
    """
    for name, param in network.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal_(
                param, gain=nn.init.calculate_gain(activation)
            )


def initialize_xavier_dynet_(model, gain=1.0):
    """
    Xaviar initialization similar to the implementation in DyNet.
    Note that in contrast to the PyTorch version, this function also randomly
    sets weights for the bias parameter.
    Parameters
    ----------
    model : nn.torch.Module
        Model to initialize the weights of
    gain : float
        See the paper. Should be 1.0 for Hyperbolic Tangent

    """
    for name, param in model.named_parameters():
        dim_len = len(param.size())
        dims = sum(param.size())
        scale = gain * math.sqrt(3 * dim_len) / math.sqrt(dims)
        torch.nn.init.uniform_(param, -scale, scale)


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze(module):
    for param in module.parameters():
        param.requires_grad = True
