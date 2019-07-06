from typing import Any

from tqdm.auto import tqdm

from parseridge.parser.training.callbacks.base_callback import Callback
from parseridge.parser.training.hyperparameters import Hyperparameters


class ProgressBarCallback(Callback):
    """
    Shows a progress bar during training.
    """

    def __init__(self, moving_average: int = 64):
        self._pbar = None
        self.template = "[{epoch:02d}/{epochs:02d}] | Batch Loss: {loss:8.4f}"

        self.prev_loss = []
        self.moving_average = moving_average

        self.batch_size = None
        self.num_epochs = None
        self.current_epoch = None

    def on_train_begin(self, epochs: int, hyper_parameters: Hyperparameters,
                       **kwargs: Any) -> None:
        self.batch_size = hyper_parameters.batch_size
        self.num_epochs = epochs

    def on_epoch_begin(self, epoch: int, num_batches: int, training_data: Any,
                       **kwargs: Any) -> None:
        self.current_epoch = epoch

        self._pbar = tqdm(total=len(training_data))

        self._pbar.set_description(self.template.format(
            epoch=self.current_epoch, epochs=self.num_epochs, loss=0
        ))

    def on_epoch_end(self, epoch_loss: float, **kwargs: Any) -> None:
        self._pbar.set_description(
            "[{epoch:02d}/{epochs:02d}] | Epoch Loss: {loss:8.4f}".format(
                epoch=self.current_epoch, epochs=self.num_epochs, loss=epoch_loss
            )
        )
        self._pbar.close()

    def on_batch_end(self, batch_loss: float, batch_data: Any, **kwargs: Any) -> None:
        if batch_loss is not None:
            self.prev_loss.append(batch_loss)
            avg_loss = sum(self.prev_loss) / len(self.prev_loss)
            self.prev_loss = self.prev_loss[-self.moving_average:]
        else:
            if self.prev_loss:
                avg_loss = sum(self.prev_loss) / len(self.prev_loss)
            else:
                avg_loss = 0

        self._pbar.set_description(
            self.template.format(
                epoch=self.current_epoch, epochs=self.num_epochs, loss=avg_loss
            )
        )

        batch_length = len(batch_data[0])
        self._pbar.update(batch_length)
