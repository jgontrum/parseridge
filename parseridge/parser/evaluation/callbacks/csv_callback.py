import csv
import os
from typing import Any, Dict, Optional

from parseridge.parser.evaluation.callbacks.base_eval_callback import EvalCallback


class CSVReporter(EvalCallback):
    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path

        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

            self.file = open(csv_path, mode="w")

            fieldnames = [
                "train_las",
                "train_uas",
                "dev_las",
                "dev_uas",
                "test_las",
                "test_uas",
                "train_loss",
            ]

            self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)

            self.writer.writeheader()

    def on_shutdown(self, **kwargs: Any) -> None:
        if self.csv_path:
            self.file.close()

    def on_eval_end(
        self, scores: Dict[str, Dict[str, float]], loss: float, **kwargs: Any
    ) -> None:
        if self.csv_path:
            self.writer.writerow(
                {
                    "train_las": scores["train"]["las"],
                    "train_uas": scores["train"]["uas"],
                    "dev_las": scores["dev"]["las"],
                    "dev_uas": scores["dev"]["uas"],
                    "test_las": scores["test"]["las"] or 0.0,
                    "test_uas": scores["test"]["uas"] or 0.0,
                    "train_loss": loss,
                }
            )
