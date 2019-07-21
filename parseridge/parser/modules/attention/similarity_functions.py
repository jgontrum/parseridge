import math
from typing import Optional

import torch
from torch import nn

from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.utils import initialize_xavier_dynet_


class DotSimilarity(Module):
    def __init__(
        self,
        key_dim: int,
        query_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def forward(self, queries, keys):
        return torch.matmul(keys, queries.unsqueeze(2))


class ScaledDotSimilarity(Module):
    def __init__(
        self,
        key_dim: int,
        query_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dot = DotSimilarity(
            key_dim=key_dim, query_dim=query_dim, value_dim=value_dim, bias=bias
        )

    def forward(self, queries, keys):
        return self.dot(queries, keys) / math.sqrt(keys.size(2))


class ConcatSimilarity(Module):
    def __init__(
        self,
        key_dim: int,
        hidden_dim: Optional[int] = None,
        query_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # This is the similarity function proposed by Bahdanau et al. (2014),
        # but simplified by Luong et al. (2015) and requires two weight matrices
        # that are learned during training.

        hidden_dim = hidden_dim if hidden_dim else key_dim
        query_dim = query_dim if query_dim else key_dim

        self.param_W = nn.Linear(query_dim + key_dim, hidden_dim, bias=bias)
        self.param_v = nn.Linear(hidden_dim, 1, bias=bias)

        initialize_xavier_dynet_(self)

    def forward(self, queries, keys):
        expanded_queries = queries.unsqueeze(1).expand(-1, keys.size(1), -1)
        concatenated = torch.cat((keys, expanded_queries), dim=2)
        return self.param_v(torch.tanh(self.param_W(concatenated)))


class GeneralSimilarity(Module):
    def __init__(
        self,
        key_dim: int,
        query_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # See Luong et al. (2015)

        key_dim = key_dim if key_dim else query_dim

        self.param_W = nn.Linear(key_dim, query_dim, bias=bias)
        self.dot = DotSimilarity(key_dim)

        initialize_xavier_dynet_(self)

    def forward(self, queries, keys):
        return self.dot(queries, self.param_W(keys))


class LearnedSimilarity(Module):
    def __init__(
        self,
        key_dim: int,
        query_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.weights = nn.Linear(in_features=key_dim, out_features=1, bias=bias)

        initialize_xavier_dynet_(self)

    def forward(self, _, keys):
        return self.weights(keys)


class DummyScoring(Module):
    """
    This function just returns "1" for every item in the sequence, making it a useful
    as a baseline where the attention mechanisms generate an unweighted average.
    """

    def __init__(
        self,
        key_dim: Optional[int] = None,
        query_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def forward(self, queries, keys):
        return torch.ones(keys.size(0), keys.size(1), 1)
