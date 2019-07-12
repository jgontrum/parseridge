from typing import Any

from parseridge.parser.training.callbacks.base_callback import Callback


class TrainSimpleLoggerCallback(Callback):
    def on_train_begin(self, **kwargs: Any) -> None:
        self.logger.info("Start training.")

    def on_train_end(self, **kwargs: Any) -> None:
        self.logger.info("Finished training.")

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        self.logger.info(f"Start epoch #{epoch + 1}.")

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        self.logger.info(f"Finished epoch #{epoch + 1}.")
