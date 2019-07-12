from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from parseridge.parser.training.callbacks.base_callback import Callback


@dataclass
class PartialFreezeEmbeddingsCallback(Callback):
    _order = 10

    freeze_indices: torch.Tensor
    embedding_layer: nn.Embedding

    def on_backward_end(self, **kwargs: Any) -> None:
        self.embedding_layer.weight.grad[self.freeze_indices] = 0.0
