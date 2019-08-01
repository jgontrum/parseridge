import torch
from torch import nn

from parseridge.parser.modules.data_parallel import Module


class PositionalEmbeddings(Module):
    def __init__(self, model_size: int, max_length: int = 1024, **kwargs):
        super().__init__(**kwargs)

        self.input_size = self.output_size = model_size

        self.embedding_size = model_size
        self.max_length = max_length
        self.padding_idx = 0

        self.emb = nn.Embedding(
            num_embeddings=self.max_length + 1,
            embedding_dim=self.embedding_size,
            padding_idx=self.padding_idx,
        )

    def forward(self, x):
        indices = torch.arange(1, x.size(1) + 1, device=self.device, dtype=torch.long)
        indices_batch = indices.expand(x.size(0), x.size(1))

        padding = torch.zeros_like(indices_batch)

        padded_indices_batch = torch.where(
            indices_batch < self.max_length, indices_batch, padding
        )

        embeddings = self.emb(padded_indices_batch)

        return x + embeddings
