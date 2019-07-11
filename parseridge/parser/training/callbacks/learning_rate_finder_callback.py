from dataclasses import dataclass, field
from typing import Any, List

from parseridge.parser.training.callbacks.base_callback import Callback, StopTraining


@dataclass
class LearningRateFinderCallback(Callback):
    _order = 100

    scheduler: Any

    smooth_loss: float = 0.05
    max_num_iterations: int = 100

    best_loss: float = float("inf")
    learning_rate_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    _num_iterations: int = 0

    def on_batch_end(self, batch_loss: float, **kwargs: Any) -> None:
        if batch_loss is None:
            return

        learning_rate = self.scheduler.get_lr()[0]
        loss = batch_loss

        if self.smooth_loss and self.loss_history:
            loss = self.smooth_loss * loss + (1 - self.smooth_loss) * self.loss_history[-1]

        if loss < self.best_loss:
            self.best_loss = loss

        self.learning_rate_history.append(learning_rate)
        self.loss_history.append(loss)

        self._num_iterations += 1

        if self._num_iterations > self.max_num_iterations:
            self.logger.info("Reached max number of iterations.")
            raise StopTraining()
