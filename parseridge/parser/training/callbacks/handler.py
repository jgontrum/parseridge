from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any, Dict

from torch.optim.optimizer import Optimizer

from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.training.callbacks.base_callback import Callback
from parseridge.utils.logger import LoggerMixin


@dataclass
class CallbackHandler(LoggerMixin):
    callbacks: List[Callback]
    model: Module
    optimizer: Optimizer

    def __post_init__(self) -> None:
        self.callbacks = self._verify_callbacks(self.callbacks)

    def register_callback(self, callback: Callback):
        self.callbacks.append(callback)
        self.callbacks = self._verify_callbacks(self.callbacks)

    @staticmethod
    def _verify_callbacks(callbacks: List[Callback]) -> List[Callback]:
        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise ValueError(
                    f"Callback {callback.__class__.__name__} is not an Callback."
                )

        return sorted(callbacks, key=lambda o: getattr(o, "_order", 0))

    def _run_callbacks(self, event: str, kwargs: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if callable(method):
                method(optimizer=self.optimizer, model=self.model, **kwargs)

    def on_train_begin(self, **kwargs: Any) -> None:
        self._run_callbacks("on_train_begin", kwargs)

    def on_epoch_begin(self, **kwargs: Any) -> None:
        self._run_callbacks("on_epoch_begin", kwargs)

    def on_batch_begin(self, **kwargs: Any) -> None:
        self._run_callbacks("on_batch_begin", kwargs)

    def on_loss_begin(self, **kwargs: Any) -> None:
        self._run_callbacks("on_loss_begin", kwargs)

    def on_backward_begin(self, **kwargs: Any) -> None:
        self._run_callbacks("on_backward_begin", kwargs)

    def on_backward_end(self, **kwargs: Any) -> None:
        self._run_callbacks("on_backward_end", kwargs)

    def on_step_end(self, **kwargs: Any) -> None:
        self._run_callbacks("on_step_end", kwargs)

    def on_batch_end(self, **kwargs: Any) -> None:
        self._run_callbacks("on_batch_end", kwargs)

    def on_epoch_end(self, **kwargs: Any) -> None:
        self._run_callbacks("on_epoch_end", kwargs)

    def on_train_end(self, **kwargs: Any) -> None:
        self._run_callbacks("on_train_end", kwargs)
