import json
import re
import time
from typing import Any, Dict, Optional

from parseridge.parser.evaluation.callbacks.base_eval_callback import EvalCallback
from parseridge.utils.google_sheets_template_engine import GoogleSheetsTemplateEngine


class EvalGoogleSheetsReporter(EvalCallback):
    _order = 20

    def __init__(
        self,
        experiment_title: str,
        sheets_id: str,
        auth_file_path: Optional[str] = None,
        hyper_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.hyper_parameters = hyper_parameters
        self.auth_file_path = auth_file_path

        if self.auth_file_path and self.hyper_parameters:
            self.hyper_parameters = hyper_parameters
            self.start_time = self.now

            self.template_engine = GoogleSheetsTemplateEngine(
                worksheet_title=experiment_title,
                sheets_id=sheets_id,
                auth_file_path=auth_file_path,
            )

            self.best_epoch = 0
            self.best_dev_las = 0
            self.epoch_start_time = self.now

            # Init static values
            self.template_engine.update_variables(
                title=experiment_title,
                start_timestamp=self.now_str,
                corpus_name=self.corpus_name,
                params_json=json.dumps(self.hyper_parameters, sort_keys=True),
                status="Running",
            )

            # Iterate over the parameters and add them to the table in multiple lines
            step = int(len(self.hyper_parameters) / 3)

            for i, item in enumerate(
                sorted(self.hyper_parameters.items(), key=lambda x: x[0])
            ):
                param_num = min(int(i / step) + 1, 3)
                self.template_engine.update_variables(
                    **{
                        f"params_{param_num}_key": item[0],
                        f"params_{param_num}_value": item[1],
                    }
                )

            # Save the changes in Google Docs
            self.template_engine.sync()

    def on_eval_end(
        self, scores: Dict[str, Dict[str, float]], loss: float, epoch: int, **kwargs: Any
    ) -> None:

        if self.auth_file_path and self.hyper_parameters:
            self.template_engine.worksheet.add_rows(1)

            if scores["dev"]["las"] > self.best_dev_las:
                self.best_dev_las = scores["dev"]["las"]
                self.best_epoch = epoch

            self.template_engine.update_variables(
                epoch=epoch,
                epoch_duration=int(self.now - self.epoch_start_time),
                epoch_loss=loss,
                epoch_training_las=scores["train"]["las"],
                epoch_training_uas=scores["train"]["uas"],
                epoch_develop_las=scores["dev"]["las"],
                epoch_develop_uas=scores["dev"]["uas"],
                epoch_test_las=scores["test"]["las"],
                epoch_test_uas=scores["test"]["uas"],
                best_epoch=self.best_epoch,
                best_dev_las=self.best_dev_las,
                duration=self.duration,
            )

            self.epoch_start_time = self.now
            self.template_engine.sync()

    def on_shutdown(self, **kwargs: Any) -> None:
        self.template_engine.update_variables(status="Finished", duration=self.duration)
        self.template_engine.sync()

    @property
    def corpus_name(self):
        return re.search(r"/(UD_.*)/", self.hyper_parameters.get("train_corpus")).group(1)

    @property
    def now_str(self):
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def now(self):
        return int(time.time())

    @property
    def duration(self):
        return self.now - self.start_time
