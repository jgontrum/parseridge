import datetime
import os
from argparse import Namespace
from time import time
from typing import Any, Dict, Optional

import yaml

from parseridge.parser.evaluation.callbacks.base_eval_callback import EvalCallback


class EvalYAMLReporter(EvalCallback):
    _order = 10

    def __init__(self, yaml_path: Optional[str] = None):
        self.yaml_path = yaml_path
        self.content = {}
        self.t0 = time()

        if self.yaml_path:
            os.makedirs(os.path.dirname(self.yaml_path), exist_ok=True)

    def _save(self):
        if self.yaml_path:
            with open(self.yaml_path, "w") as f:
                yaml.safe_dump(self.content, f)

    def on_initialization(self, cli_args: Optional[Namespace], **kwargs: Any) -> None:
        self.t0 = time()
        self.content["start_time"] = datetime.datetime.now().isoformat()
        self.content["epochs"] = {}

        if cli_args:
            self.content["parameters"] = vars(cli_args)

        self._save()

    def on_shutdown(self, **kwargs: Any) -> None:
        self.content["end_time"] = datetime.datetime.now().isoformat()
        self.content["duration"] = time() - self.t0
        self._save()

    def on_eval_end(
        self, scores: Dict[str, Dict[str, float]], loss: float, epoch: int, **kwargs: Any
    ) -> None:
        self.content["epochs"][epoch] = {
            "epoch": epoch,
            "train_las": scores["train"]["las"],
            "train_uas": scores["train"]["uas"],
            "train_others": scores["train"]["all"],
            "dev_las": scores["dev"]["las"],
            "dev_uas": scores["dev"]["uas"],
            "dev_others": scores["dev"]["all"],
            "test_las": scores["test"]["las"] or 0.0,
            "test_uas": scores["test"]["uas"] or 0.0,
            "test_others": scores["test"]["all"] or {},
            "train_loss": loss,
        }

        self._save()
