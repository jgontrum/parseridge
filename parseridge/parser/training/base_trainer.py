from abc import ABC, abstractmethod
from typing import List

import torch
from torch.optim.optimizer import Optimizer

from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.training.callbacks.base_callback import Callback
from parseridge.parser.training.callbacks.handler import CallbackHandler
from parseridge.parser.training.callbacks.model_training_callback import (
    ModelTrainingCallback,
)
from parseridge.utils.logger import LoggerMixin


class Trainer(LoggerMixin, ABC):
    def __init__(
        self, model: Module, optimizer: Optimizer, callbacks: List[Callback] = None
    ):
        self.model = model
        self.optimizer = optimizer

        self.callback_handler = CallbackHandler(
            callbacks=callbacks or [], model=self.model, optimizer=self.optimizer
        )

        self.callback_handler.register_callback(ModelTrainingCallback())

        self.last_epoch = 0

    @abstractmethod
    def fit(self, epochs: int):
        pass

    def fit_one_cycle(self):
        self.fit(epochs=1)

    def learn(self, loss: torch.Tensor):
        self.callback_handler.on_loss_begin(loss=loss)

        # Compute the gradients
        self.callback_handler.on_backward_begin()

        loss.backward()

        self.callback_handler.on_backward_end()

        # Update the weights
        self.optimizer.step()

        self.callback_handler.on_step_end()

        # Clear all previous gradients
        self.optimizer.zero_grad()
