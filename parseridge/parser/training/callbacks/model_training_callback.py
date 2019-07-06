from typing import Any

from torch.optim.optimizer import Optimizer

from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.training.callbacks.base_callback import Callback


class ModelTrainingCallback(Callback):
    """
    This is a basic callback that ensures that the model is set to training mode in the
    beginning of an epoch and that all the gradients are cleared.
    """

    def on_epoch_begin(self, optimizer: Optimizer, model: Module, **kwargs: Any) -> None:
        optimizer.zero_grad()
        model.train()

    def on_epoch_end(self, optimizer: Optimizer, model: Module, **kwargs: Any) -> None:
        optimizer.zero_grad()
        model.eval()

    def on_batch_begin(self, model: Module, **kwargs: Any) -> None:
        # If the model implements a 'reset_' method, call it.
        if hasattr(model, "reset_") and callable(model.reset_):
            model.reset_()
