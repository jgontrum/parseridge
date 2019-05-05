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
    if isinstance(model, torch.nn.Parameter):
        dim_len = len(model.size())
        dims = sum(model.size())
        scale = gain * math.sqrt(3 * dim_len) / math.sqrt(dims)
        torch.nn.init.uniform_(model, -scale, scale)

    else:
        for name, param in model.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
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


def add_padding(sequence, length, padding_item):
    assert len(sequence) <= length

    while len(sequence) < length:
        sequence.append(padding_item)

    return sequence


def create_mask(lengths, max_len=None, device="cpu"):
    if not max_len:
        max_len = max(lengths)

    # Build a mask that has a 0 for words and a 1 for the padding.
    mask = [
        [0] * length + [1] * (max_len - length)
        for length in lengths
    ]

    # Create a ByteTensor on the given device and return it.
    return torch.tensor(mask, device=device, dtype=torch.uint8)


def lookup_tensors_for_indices(indices_batch, sequence_batch, padding, size):
    size = max(1, size)
    batch = []
    for index_list, lstm_out in zip(indices_batch, sequence_batch):
        items = [lstm_out[i] for i in index_list]
        while len(items) < size:
            items.append(padding)

        # Turn list of tensors into one tensor
        items = torch.stack(items)

        # Flatten the tensor
        # items = items.view((-1,)).contiguous()

        batch.append(items)

    return torch.stack(batch).contiguous()
