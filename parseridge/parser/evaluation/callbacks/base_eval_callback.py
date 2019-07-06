from abc import ABC
from typing import Any

from parseridge.utils.logger import LoggerMixin


class EvalCallback(LoggerMixin, ABC):
    _order = 0

    def on_initialization(self, **kwargs: Any) -> None:
        pass

    def on_eval_begin(self, **kwargs: Any) -> None:
        pass

    def on_epoch_begin(self, **kwargs: Any) -> None:
        pass

    def on_batch_begin(self, **kwargs: Any) -> None:
        pass

    def on_batch_end(self, **kwargs: Any) -> None:
        pass

    def on_epoch_end(self, **kwargs: Any) -> None:
        pass

    def on_eval_end(self, **kwargs: Any) -> None:
        pass

    def on_shutdown(self, **kwargs: Any) -> None:
        pass
