import os
from typing import Any, Optional

import torch

from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.training.callbacks.base_callback import Callback


class SaveModelCallback(Callback):
    _order = 5

    def __init__(self, folder_path: Optional[str] = None):
        self.folder = folder_path
        if self.folder:
            os.makedirs(folder_path, exist_ok=True)

    def on_epoch_end(self, epoch: int, model: Module, **kwargs: Any) -> None:
        if self.folder:
            file_name = f"{self.folder}/epoch_{epoch + 1}.torch"
            torch.save(model.state_dict(), file_name)
