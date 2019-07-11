from typing import Any

from parseridge.parser.training.callbacks.base_callback import Callback


class EvaluationCallback(Callback):
    """
    This is a basic callback that ensures that the model is set to training mode in the
    beginning of an epoch and that all the gradients are cleared.
    """

    _order = 100

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def on_train_end(self, **kwargs: Any) -> None:
        self.evaluator.shutdown()

    def on_epoch_end(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:
        self.evaluator.evaluate(epoch=epoch, loss=epoch_loss)
