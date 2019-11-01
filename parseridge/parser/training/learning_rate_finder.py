import math
from dataclasses import dataclass
from typing import Union, Optional

from parseridge.corpus.corpus import Corpus
from parseridge.corpus.training_data import ConLLDataset
from parseridge.parser.training.base_trainer import Trainer
from parseridge.parser.training.callbacks.learning_rate_finder_callback import (
    LearningRateFinderCallback,
)
from parseridge.parser.training.callbacks.lr_scheduler_callback import LRSchedulerCallback
from parseridge.parser.training.callbacks.progress_bar_callback import ProgressBarCallback
from parseridge.parser.training.cyclic_lr import CyclicLR
from parseridge.parser.training.hyperparameters import Hyperparameters
from parseridge.utils.logger import LoggerMixin


@dataclass
class LearningRateFinder(LoggerMixin):
    trainer: Trainer

    min_learning_rate: float = 0.001
    max_learning_rate: float = 10.0
    num_iterations: int = 100
    divergence_threshold: float = 5.0
    mode: str = "triangular"  # One of {triangular, triangular2, exp_range}

    def find_learning_rate(
        self,
        training_data: Union[Corpus, ConLLDataset],
        hyper_parameters: Optional[Hyperparameters] = None,
        **kwargs,
    ):
        step_size = int(math.floor(len(training_data) / hyper_parameters.batch_size / 2))

        scheduler = CyclicLR(
            optimizer=self.trainer.optimizer,
            base_lr=self.min_learning_rate,
            max_lr=self.max_learning_rate,
            step_size_up=step_size,
            mode=self.mode,
        )

        lr_finder_callback = LearningRateFinderCallback(scheduler=scheduler)

        training_callbacks = [
            lr_finder_callback,
            LRSchedulerCallback(scheduler, when="after_epoch"),
            ProgressBarCallback(moving_average=64),
        ]

        self.trainer.register_callbacks(training_callbacks)

        self.trainer.fit(
            epochs=10, training_data=training_data, hyper_parameters=hyper_parameters
        )
