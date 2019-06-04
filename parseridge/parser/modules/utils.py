import math

import torch
from torch import nn
from torch.nn.functional import pad


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

    def set_weights(param):
        dim_len = len(param.size())
        dims = sum(param.size())
        scale = gain * math.sqrt(3 * dim_len) / math.sqrt(dims)
        torch.nn.init.uniform_(param, -scale, scale)

    if isinstance(model, torch.nn.Parameter):
        set_weights(model)

    else:
        for name, param in model.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                set_weights(param)


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


def pad_tensor(tensor, length, padding=0):
    tensor_length = tensor.shape[0]

    assert length >= tensor_length, "Tensor too long to pad."
    assert len(tensor.shape) == 1, "Tensor must be one-dimensional."

    return pad(tensor, [0, length - tensor_length], value=padding)


def pad_tensor_list(tensors, padding=0):
    max_length = max([len(tensor) for tensor in tensors])
    padded_tensors = [
        pad_tensor(tensor, length=max_length, padding=padding) for tensor in tensors
    ]

    return torch.stack(padded_tensors)


def create_mask(lengths, max_len=None, device="cpu"):
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.cpu().tolist()

    if not max_len:
        max_len = max(lengths)

    # Build a mask that has a 0 for words and a 1 for the padding.
    mask = [
        [0] * length + [1] * (max_len - length)
        for length in lengths
    ]

    # Create a ByteTensor on the given device and return it.
    return torch.tensor(mask, device=device, dtype=torch.uint8)


def lookup_tensors_for_indices(indices_batch, sequence_batch):
    batch = []
    for index_list, lstm_out in zip(indices_batch, sequence_batch):
        items = torch.index_select(lstm_out, dim=0, index=index_list)
        batch.append(items)

    return torch.stack(batch).contiguous()
