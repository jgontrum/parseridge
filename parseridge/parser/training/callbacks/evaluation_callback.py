from typing import Any

from parseridge.parser.training.callbacks.base_callback import Callback


class EvaluationCallback(Callback):
    """
    This is a basic callback that ensures that the model is set to training mode in the
    beginning of an epoch and that all the gradients are cleared.
    """
    _order = 100

    def __init__(self, evaluator: "parseridge.parser.evaluation.Evaluator"):
        self._current_epoch = None
        self.evaluator = evaluator

    def on_train_end(self, **kwargs: Any) -> None:
        self.evaluator.shutdown()

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        self._current_epoch = epoch

    def on_epoch_end(self, **kwargs: Any) -> None:
        self.evaluator.evaluate(epoch=self._current_epoch)
