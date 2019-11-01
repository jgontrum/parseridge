from dataclasses import dataclass
from typing import List, Any, Dict

from parseridge.parser.evaluation.callbacks.base_eval_callback import EvalCallback
from parseridge.utils.logger import LoggerMixin


@dataclass
class EvalCallbackHandler(LoggerMixin):
    callbacks: List[EvalCallback]

    def __post_init__(self) -> None:
        self.callbacks = self._verify_callbacks(self.callbacks)

    def register_callback(self, callback: EvalCallback):
        self.callbacks.append(callback)
        self.callbacks = self._verify_callbacks(self.callbacks)

    @staticmethod
    def _verify_callbacks(callbacks: List[EvalCallback]) -> List[EvalCallback]:
        new_callbacks = []
        for callback in callbacks:
            if callback is None:
                continue

            if not isinstance(callback, EvalCallback):
                raise ValueError(
                    f"Callback {callback.__class__.__name__} is not an EvalCallback."
                )

            new_callbacks.append(callback)

        return sorted(new_callbacks, key=lambda o: getattr(o, "_order", 0))

    def _run_callbacks(self, event: str, kwargs: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if callable(method):
                method(**kwargs)

    def on_initialization(self, **kwargs: Any) -> None:
        self._run_callbacks("on_initialization", kwargs)

    def on_eval_begin(self, **kwargs: Any) -> None:
        self._run_callbacks("on_eval_begin", kwargs)

    def on_epoch_begin(self, **kwargs: Any) -> None:
        self._run_callbacks("on_epoch_begin", kwargs)

    def on_batch_begin(self, **kwargs: Any) -> None:
        self._run_callbacks("on_batch_begin", kwargs)

    def on_batch_end(self, **kwargs: Any) -> None:
        self._run_callbacks("on_batch_end", kwargs)

    def on_epoch_end(self, **kwargs: Any) -> None:
        self._run_callbacks("on_epoch_end", kwargs)

    def on_eval_end(self, **kwargs: Any) -> None:
        self._run_callbacks("on_eval_end", kwargs)

    def on_shutdown(self, **kwargs: Any) -> None:
        self._run_callbacks("on_shutdown", kwargs)
