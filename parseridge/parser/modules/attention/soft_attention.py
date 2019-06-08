from typing import Tuple

import torch
from torch import Tensor

from parseridge.parser.modules.attention.similarity_functions import DotSimilarity, \
    GeneralSimilarity, ScaledDotSimilarity, ConcatSimilarity
from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.utils import mask_


class Attention(Module):

    def __init__(self, similarity="dot", normalization="softmax", device="cpu",
                 **kwargs):
        super(Attention, self).__init__()
        self.device = device

        similarity_functions = {
            "dot": lambda kwargs: DotSimilarity(**kwargs),
            "scaled_dot": lambda kwargs: ScaledDotSimilarity(**kwargs),
            "general": lambda kwargs: GeneralSimilarity(**kwargs),
            "concat": lambda kwargs: ConcatSimilarity(**kwargs),
        }

        normalization_functions = {
            "softmax": lambda t: torch.nn.functional.softmax(t, dim=1),
            "sigmoid": lambda t: torch.sigmoid(t),
            "identity": lambda t: t
        }

        self.similarity_function = similarity_functions[similarity](kwargs)
        self.normalize = normalization_functions[normalization]

    def forward(self, queries: Tensor, keys: Tensor, sequence_lengths: Tensor,
                values: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        if values is None:
            values = keys

        # Compare keys to queries
        attention_logits = self.similarity_function(queries, keys)

        # Mask scores for padding keys
        attention_logits = mask_(attention_logits, sequence_lengths, device=self.device)

        # Apply normalization function (e.g. softmax)
        attention_energies = self.normalize(attention_logits)

        # Multiply the values with the attention scores
        weighted_values = values * attention_energies

        # Compute a weighted average to get a sequence encoding
        context_vector = torch.sum(weighted_values, dim=1)

        return context_vector, weighted_values, attention_energies
