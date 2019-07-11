from typing import Any

from parseridge.parser.training.callbacks.base_callback import Callback


class LRSchedulerCallback(Callback):
    _order = 10

    def __init__(self, scheduler: Any, when: str = "after_epoch"):
        assert when in ["after_epoch", "after_batch"], f"'{when}' not valid."

        self.when = when
        self.scheduler = scheduler

    def on_epoch_end(self, **kwargs: Any) -> None:
        if self.when == "after_epoch":
            self.scheduler.step()

    def on_batch_end(self, **kwargs: Any) -> None:
        if self.when == "after_batch":
            self.scheduler.step()
