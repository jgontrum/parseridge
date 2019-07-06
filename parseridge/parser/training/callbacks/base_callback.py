from abc import ABC
from typing import Any

from parseridge.utils.logger import LoggerMixin


class Callback(LoggerMixin, ABC):
    """
    Base class for callbacks that want to record values,
    dynamically change learner params, etc.
    """
    _order = 0

    def on_train_begin(self, **kwargs: Any) -> None:
        """"To initialize constants in the callback."""
        pass

    def on_epoch_begin(self, **kwargs: Any) -> None:
        """At the beginning of each epoch."""
        pass

    def on_batch_begin(self, **kwargs: Any) -> None:
        """Set HP before the output and loss are computed."""
        pass

    def on_loss_begin(self, **kwargs: Any) -> None:
        """Called after forward pass but before loss has been computed."""
        pass

    def on_backward_begin(self, **kwargs: Any) -> None:
        """
        Called after the forward pass and the loss has been computed, but before backprop.
        """
        pass

    def on_backward_end(self, **kwargs: Any) -> None:
        """Called after backprop but before optimizer step.
        Useful for true weight decay in AdamW.
        """
        pass

    def on_step_end(self, **kwargs: Any) -> None:
        """Called after the step of the optimizer but before the gradients are zeroed."""
        pass

    def on_batch_end(self, **kwargs: Any) -> None:
        """Called at the end of the batch."""
        pass

    def on_epoch_end(self, **kwargs: Any) -> None:
        """Called at the end of an epoch."""
        pass

    def on_train_end(self, **kwargs: Any) -> None:
        """Useful for cleaning up things and saving files/models."""
        pass
