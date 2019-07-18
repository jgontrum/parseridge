from typing import Tuple, Optional

import torch
from torch import Tensor, nn

from parseridge.parser.modules.attention.similarity_functions import (
    DotSimilarity,
    GeneralSimilarity,
    ScaledDotSimilarity,
    ConcatSimilarity,
    LearnedSimilarity,
)
from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.utils import mask_


class Attention(Module):
    SCORING_FUNCTIONS = {
        "dot": lambda kwargs: DotSimilarity(**kwargs),
        "scaled_dot": lambda kwargs: ScaledDotSimilarity(**kwargs),
        "general": lambda kwargs: GeneralSimilarity(**kwargs),
        "concat": lambda kwargs: ConcatSimilarity(**kwargs),
        "learned": lambda kwargs: LearnedSimilarity(**kwargs),
    }

    NORMALIZATION_FUNCTIONS = {
        "softmax": lambda t: torch.nn.functional.softmax(t, dim=1),
        "sigmoid": lambda t: torch.sigmoid(t),
        "identity": lambda t: t,
    }

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: Optional[int] = None,
        query_output_dim: Optional[int] = None,
        key_output_dim: Optional[int] = None,
        value_output_dim: Optional[int] = None,
        similarity="dot",
        normalization="softmax",
        device="cpu",
        **kwargs,
    ):
        super().__init__(device=device)
        self.input_size = key_dim
        self.output_size = key_dim if not key_output_dim else key_output_dim

        if not value_dim:
            value_dim = key_dim

        # Add input transformation layers if required
        if query_output_dim:
            self.query_transform = nn.Linear(
                in_features=query_dim, out_features=query_output_dim
            )

        if key_output_dim:
            self.key_transform = nn.Linear(in_features=key_dim, out_features=key_output_dim)

        if value_output_dim:
            self.value_transform = nn.Linear(
                in_features=value_dim, out_features=value_output_dim
            )

        assert key_output_dim == value_output_dim

        kwargs.update(
            {
                "query_dim": query_dim if not query_output_dim else query_output_dim,
                "key_dim": key_dim if not key_output_dim else key_output_dim,
                "value_dim": value_dim if not value_output_dim else value_output_dim,
            }
        )

        self.similarity_function = self.SCORING_FUNCTIONS[similarity](kwargs)
        self.normalize = self.NORMALIZATION_FUNCTIONS[normalization]

    def forward(
        self, queries: Tensor, keys: Tensor, sequence_lengths: Tensor, values: Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if keys.size(1) == 0:
            # The whole batch contains only empty sequences
            dummy_context_vector = torch.zeros(
                (keys.size(0), self.output_size), device=self.device
            )

            dummy_weighted_values = torch.zeros(
                (keys.size(0), 1, self.output_size), device=self.device
            )

            dummy_attention_energies = torch.zeros((keys.size(0), 1, 1), device=self.device)

            return dummy_context_vector, dummy_weighted_values, dummy_attention_energies

        if values is None:
            values = keys

        queries = self._transform_query(queries)
        keys = self._transform_keys(keys)
        values = self._transform_values(values)

        # Compare keys to queries
        attention_logits = self.similarity_function(queries, keys)

        # Mask scores for padding keys
        attention_logits = mask_(attention_logits, sequence_lengths, device=self.device)

        # Apply normalization function (e.g. softmax)
        attention_energies = self.normalize(attention_logits)

        # Replace nan with zeros. Happens when a sequence length is 0.
        attention_energies = torch.where(
            torch.isnan(attention_energies),  # Condition
            torch.zeros_like(attention_energies),  # Use this value if condition is true
            attention_energies,  # Use this value if condition is false
        )

        # Multiply the values with the attention scores
        weighted_values = values * attention_energies

        # Compute a weighted average to get a sequence encoding
        context_vector = torch.sum(weighted_values, dim=1)

        return context_vector, weighted_values, attention_energies

    def _transform_query(self, query: Tensor) -> Tensor:
        if hasattr(self, "query_transform"):
            return torch.tanh(self.query_transform(query))
        return query

    def _transform_keys(self, keys: Tensor) -> Tensor:
        if hasattr(self, "key_transform"):
            return torch.tanh(self.key_transform(keys))
        return keys

    def _transform_values(self, values: Tensor) -> Tensor:
        if hasattr(self, "value_transform"):
            return torch.tanh(self.value_transform(values))
        return values
