import math

import torch
from torch import nn

from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.utils import initialize_xavier_dynet_


class DotSimilarity(Module):

    def __init__(self, **kwargs):
        super(DotSimilarity, self).__init__()

    def forward(self, queries, keys):
        return torch.matmul(keys, queries.unsqueeze(2))


class ScaledDotSimilarity(Module):

    def __init__(self, **kwargs):
        super(ScaledDotSimilarity, self).__init__()
        self.dot = DotSimilarity()

    def forward(self, queries, keys):
        return self.dot(queries, keys) / math.sqrt(keys.size(2))


class ConcatSimilarity(Module):

    def __init__(self, query_dim: int, key_dim: int = None, hidden_dim: int = None,
                 bias=False, **kwargs):
        super(ConcatSimilarity, self).__init__()
        # This is the similarity function proposed by Bahdanau et al. (2014),
        # but simplified by Luong et al. (2015) and requires two weight matrices
        # that are learned during training.

        hidden_dim = hidden_dim if hidden_dim else query_dim
        key_dim = key_dim if key_dim else query_dim

        self.param_W = nn.Linear(query_dim + key_dim, hidden_dim, bias=bias)
        self.param_v = nn.Linear(hidden_dim, 1, bias=bias)

        initialize_xavier_dynet_(self)

    def forward(self, queries, keys):
        expanded_queries = queries.unsqueeze(1).expand(-1, keys.size(1), -1)
        concatenated = torch.cat((keys, expanded_queries), dim=2)
        return self.param_v(torch.tanh(self.param_W(concatenated)))


class GeneralSimilarity(Module):

    def __init__(self, query_dim: int, key_dim: int = None, bias=False, **kwargs):
        super(GeneralSimilarity, self).__init__()
        # See Luong et al. (2015)

        key_dim = key_dim if key_dim else query_dim

        self.param_W = nn.Linear(key_dim, query_dim, bias=bias)
        self.dot = DotSimilarity()

        initialize_xavier_dynet_(self)

    def forward(self, queries, keys):
        return self.dot(queries, self.param_W(keys))
