from typing import List

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from parseridge.parser.modules.data_parallel import Module


class RNN(Module):
    """
    Wrapper around a PyTorch RNN to include packing batch inputs and sorting the
    input by length.
    """

    def __init__(self, rnn: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.rnn = rnn

    def forward(self, input: torch.Tensor, sequence_lengths: List) -> torch.Tensor:
        seq_lengths = torch.tensor(
            [max(1, sequence_len) for sequence_len in sequence_lengths],
            device=self.device,
            dtype=torch.long,
        )

        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        seq_tensors = input[perm_idx]

        packed_input = pack_padded_sequence(seq_tensors, seq_lengths, batch_first=True)

        packed_output, _ = self.rnn(packed_input)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        output_original_order = output[perm_idx]
        return output_original_order
