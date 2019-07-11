import math

import torch

from parseridge.parser.modules.data_parallel import Module


# Implementation inspired by
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
class PositionalEncoder(Module):
    def __init__(self, model_size=128, max_length=200, **kwargs):
        super().__init__(**kwargs)
        self.model_size = self.input_size = self.output_size = model_size

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_length, self.model_size)
        for pos in range(max_length):
            for i in range(0, self.model_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.model_size)))
                pe[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / self.model_size))
                )

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # make embeddings relatively larger
        x *= math.sqrt(self.model_size)

        # add constant to embedding
        seq_len = x.size(1)

        pe = self.pe.clone().detach().requires_grad_(False)
        return x + pe[:, :seq_len]
