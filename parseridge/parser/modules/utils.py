import math
from typing import Any, List

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
        if "bias" in name:
            nn.init.constant_(param, 0.0)
        elif "weight" in name:
            nn.init.xavier_normal_(param, gain=nn.init.calculate_gain(activation))


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
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                set_weights(param)


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze(module):
    for param in module.parameters():
        param.requires_grad = True


def pad_list(sequence: List, length: int, padding: Any):
    assert len(sequence) <= length
    return sequence + (length - len(sequence)) * [padding]


def pad_list_of_lists(sequences: List[List], padding: Any = 0):
    max_length = max([len(sequence) for sequence in sequences])

    return [
        pad_list(sequence, length=max_length, padding=padding) for sequence in sequences
    ]


def pad_tensor(tensor, length, padding=0):
    tensor_length = tensor.shape[0]

    assert length >= tensor_length, "Tensor too long to pad."
    assert len(tensor.shape) == 1, "Tensor must be one-dimensional."

    return pad(tensor, [0, length - tensor_length], value=padding)


def pad_tensor_list(tensors, padding=0, length=None):
    max_length = max([len(tensor) for tensor in tensors])
    if length is not None:
        max_length = max(max_length, length)

    padded_tensors = [
        pad_tensor(tensor, length=max_length, padding=padding) for tensor in tensors
    ]

    return torch.stack(padded_tensors)


def lookup_tensors_for_indices(indices_batch, sequence_batch):
    return torch.stack(
        [
            torch.index_select(batch, dim=0, index=indices)
            for batch, indices in zip(sequence_batch, indices_batch)
        ]
    )


def get_padded_tensors_for_indices(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    contextualized_input_batch: torch.Tensor,
    max_length: int,
    padding: torch.Tensor,
    device: str = "cpu",
):
    indices = pad_tensor_list(indices, length=max_length)
    # Lookup the contextualized tokens from the indices
    batch = lookup_tensors_for_indices(indices, contextualized_input_batch)

    batch_size = batch.size(0)
    sequence_size = max(batch.size(1), max_length)
    token_size = batch.size(2)

    # Expand the padding vector over the size of the batch
    padding_batch = padding.expand(batch_size, sequence_size, token_size)

    if max(lengths) == 0:
        # If the batch is completely empty, we can just return the whole padding batch
        batch_padded = padding_batch
    else:
        # Build a mask and expand it over the size of the batch
        mask = torch.arange(sequence_size, device=device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(2).expand(batch_size, sequence_size, token_size)

        batch_padded = torch.where(
            mask,  # Condition
            batch,  # If condition is 1
            padding_batch,  # If condition is 0
        )

        # Cut the tensor at the specified length
        batch_padded = torch.split(batch_padded, max_length, dim=1)[0]

    # Flatten the output by concatenating the token embeddings
    return batch_padded.contiguous().view(batch_padded.size(0), -1)


def get_mask(batch, lengths, device="cpu"):
    max_len = batch.size(1)
    return torch.arange(max_len, device=device)[None, :] < lengths[:, None]


def mask_(batch, lengths, masked_value=float("-inf"), device="cpu"):
    mask = get_mask(batch, lengths, device)
    batch[~mask] = masked_value
    return batch


def to_int_tensor(data: Any, device="cpu"):
    if isinstance(data, torch.Tensor):
        return data.type(torch.int64).to(device=device)
    return torch.tensor(data, dtype=torch.int64, device=device)


def to_byte_tensor(data: Any, device="cpu"):
    if isinstance(data, torch.Tensor):
        return data.type(torch.uint8).to(device=device)
    return torch.tensor(data, dtype=torch.uint8, device=device)


def to_float_tensor(data: Any, device="cpu"):
    if isinstance(data, torch.Tensor):
        return data.type(torch.float32).to(device=device)
    return torch.tensor(data, dtype=torch.float32, device=device)
