from typing import Any

from tqdm.auto import tqdm

from parseridge.parser.evaluation.callbacks.base_eval_callback import EvalCallback


class EvalProgressBarCallback(EvalCallback):
    """
    Shows a progress bar during prediction.
    """

    def __init__(self):
        self._pbar = None
        self.batch_size = None

    def on_epoch_begin(self, dataset: Any, corpus_type: str, **kwargs: Any) -> None:
        self._pbar = tqdm(
            total=len(dataset), desc=f"Predicting '{corpus_type}'", leave=False)

    def on_epoch_end(self, **kwargs: Any) -> None:
        self._pbar.close()

    def on_batch_end(self, batch_data: Any, **kwargs: Any) -> None:
        batch_length = len(batch_data[0])
        self._pbar.update(batch_length)
