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

    def forward(self, input: torch.Tensor, sequences: List,
                ignore_empty_sequences=False) -> torch.Tensor:
        original_sequence_lengths = [len(sequence) for sequence in sequences]

        # The batch must be ordered in decreasing order. Save the original order here.
        order, _ = zip(*sorted(
            enumerate(original_sequence_lengths),
            key=lambda x: x[1],
            reverse=True
        ))

        sequences_in_order = all(order[i] <= order[i+1] for i in range(len(order)-1))

        # If we do not want to process empty sequences, we have to filter them out of
        # the batch.
        num_empty_sequences = original_sequence_lengths.count(0)
        if num_empty_sequences and ignore_empty_sequences:
            ignored_indices = order[-num_empty_sequences:]
            lookup_indices = order[:-num_empty_sequences]
        else:
            ignored_indices = []
            lookup_indices = order

        # In case there only empty sequences and we choose to ignore them, simply return
        # the input. There is no work to be done here.
        if not lookup_indices:
            return input

        # Sort the sequences if needed
        if not sequences_in_order or ignored_indices:
            lookup_indices_tensor = torch.tensor(
                lookup_indices, device=self.device, dtype=torch.long)
            sorted_input = torch.index_select(input, dim=0, index=lookup_indices_tensor)
        else:
            sorted_input = input

        # Create a list of sequence lengths for the sentences in the new order.
        # If we do not ignore empty sequences, we set their size to 1.
        sequence_lengths = [
            len(sequences[i]) if ignore_empty_sequences else max(1, len(sequences[i]))
            for i in lookup_indices
        ]

        # Mask padding in the batch by using a PackedSequence
        packed_input = pack_padded_sequence(
            sorted_input,
            torch.tensor(sequence_lengths, dtype=torch.int64, device=self.device),
            batch_first=True
        )

        # Run the RNN
        packed_outputs, _ = self.rnn(packed_input)

        # Extract the masked output again
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        # If we excluded some indices from the RNN, we include the original sequences now
        if ignored_indices:
            ignored_indices_tensor = torch.tensor(
                ignored_indices, device=self.device, dtype=torch.long)
            ignored_input = torch.index_select(input, dim=0, index=ignored_indices_tensor)
            outputs = torch.cat((outputs, ignored_input), dim=0)

        if not sequences_in_order or ignored_indices:
            # Bring the sequences back into the original order and return them
            order_tensor = torch.tensor(order, device=self.device, dtype=torch.long)
            return torch.index_select(outputs, dim=0, index=order_tensor)
        else:
            return outputs
