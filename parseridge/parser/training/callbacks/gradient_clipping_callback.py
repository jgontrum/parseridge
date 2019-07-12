from dataclasses import dataclass
from typing import Any

import torch

from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.training.callbacks.base_callback import Callback


@dataclass
class GradientClippingCallback(Callback):
    threshold: float = 10.0

    def on_backward_end(self, model: Module, **kwargs: Any) -> None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.threshold)
