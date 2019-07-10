from copy import deepcopy
from dataclasses import dataclass

from parseridge.utils.logger import LoggerMixin

"""
TODO
[ ] Group the parameters
[ ] Add save to / load from YAML
[x] Add overwrite method from kwargs
"""


@dataclass
class Hyperparameters(LoggerMixin):
    """
    Container for the various hyper-parameters used in the training process.
    They are stored here to keep the code in the trainer clean.
    """

    learning_rate: float = 1e-3

    batch_size: int = 4
    error_probability: float = 0.1
    oov_probability: float = 0.25
    margin_threshold: float = 2.5
    token_dropout: float = 0.01

    loss_function: str = "CrossEntropy"  # See Criterion.LOSS_FUNCTIONS

    def update(self, **kwargs):
        new_object = deepcopy(self)

        for parameter_name, value in kwargs.items():
            if not parameter_name.startswith("_") and hasattr(new_object, parameter_name):
                setattr(new_object, parameter_name, value)
            else:
                self.logger.warning(f"Cannot update value for '{parameter_name}'.")

        return new_object
