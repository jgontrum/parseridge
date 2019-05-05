import torch
from torch import nn

from parseridge.parser.modules.data_parallel import Module


class PositionalEmbeddings(Module):

    def __init__(self, embedding_size=128, max_length=80, **kwargs):
        super().__init__(**kwargs)

        self.embedding_size = embedding_size
        self.max_length = max_length
        self.padding_idx = 0
        self.output_size = self.embedding_size

        self.emb = nn.Embedding(
            num_embeddings=self.max_length + 1,
            embedding_dim=self.embedding_size,
            padding_idx=self.padding_idx,
        )

    def forward(self, input_lengths):
        max_length = max(1, max(input_lengths))

        position_batch = []
        for length in input_lengths:
            positions = list(range(1, length + 1))
            positions = [min(self.max_length, position) for position in positions]

            padding = [self.padding_idx] * (max_length - length)
            position_batch.append(positions + padding)

        position_batch = torch.tensor(
            position_batch,
            device=self.device,
            dtype=torch.long
        )

        return self.emb(position_batch)
