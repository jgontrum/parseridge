import math

import numpy as np
import torch
from torch import nn

from parseridge.parser.modules.data_parallel import Module


class PositionalEncoder(Module):
    def __init__(
        self, model_size: int = 128, max_length: int = 200, dropout: float = 0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.model_size = self.input_size = self.output_size = model_size

        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * i / model_size) for i in range(model_size)]
                for pos in range(1, max_length + 1)
            ]
        )

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        self.pe = torch.from_numpy(position_enc).float().to(self.device)
        self.pe = self.pe.requires_grad_(False)

        self.norm = nn.LayerNorm(self.model_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if x.size(1) == 0:
            return x

        # make embeddings relatively larger
        x *= math.sqrt(self.model_size)

        pe = self.pe[: x.size(1)]
        return self.norm(self.dropout(x + pe))
