from typing import Tuple, Optional

import torch
from torch import nn, Tensor

from parseridge.parser.modules.attention.soft_attention import Attention
from parseridge.parser.modules.utils import initialize_xavier_dynet_, mask_


class UniversalAttention(Attention):

    def __init__(self, query_dim: int, query_output_dim: Optional[int] = None, **kwargs):
        super().__init__(
            query_dim=query_dim, key_dim=query_dim, query_output_dim=query_output_dim,
            key_output_dim=query_output_dim, **kwargs)

        self.query_param = nn.Parameter(torch.rand(query_dim))

    def forward(self, keys: Tensor, sequence_lengths: Tensor,
                values: Tensor = None, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        queries = self.query_param.expand(keys.size(0), -1)
        return super().forward(queries, keys, sequence_lengths, values)


class LinearAttention(Attention):
    def __init__(self, query_dim: int, query_output_dim: Optional[int] = None, **kwargs):
        super().__init__(
            query_dim=query_dim, key_dim=query_dim, query_output_dim=query_output_dim,
            key_output_dim=query_output_dim, **kwargs)

        self.learn_input = nn.Sequential(
            nn.Linear(
                in_features=query_dim,
                out_features=query_dim
            ),
            nn.Tanh()
        )

        self.similarity_function = nn.Linear(
            in_features=query_dim,
            out_features=1
        )

        initialize_xavier_dynet_(self)

    def forward(self, keys: Tensor, sequence_lengths: Tensor,
                values: Tensor = None, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        if values is None:
            values = keys

        keys = self.learn_input(keys)

        # Compare keys to queries
        attention_logits = self.similarity_function(keys)

        # Mask scores for padding keys
        attention_logits = mask_(attention_logits, sequence_lengths, device=self.device)

        # Apply normalization function (e.g. softmax)
        attention_energies = self.normalize(attention_logits)

        # Multiply the values with the attention scores
        weighted_values = values * attention_energies

        # Compute a weighted average to get a sequence encoding
        context_vector = torch.sum(weighted_values, dim=1)

        return context_vector, weighted_values, attention_energies
