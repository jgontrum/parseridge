import json
import re
import subprocess
import time
from abc import ABC, abstractmethod
from random import randint

from parseridge.utils.google_sheets_template_engine import GoogleSheetsTemplateEngine
from parseridge.utils.logger import LoggerMixin


class BaseReporter(LoggerMixin, ABC):

    def __init__(self, hyper_parameters: dict):
        self.start_time = self.now
        self.hyper_parameters = hyper_parameters

        self.git_commit = self.git_branch = self.git_diff = "No Git available"

        self.last_epoch_start = self.start_time
        self.best_result = {
            "epoch": 0, "value": float("-inf")
        }

    def _set_git_info(self):
        try:
            self.git_commit = subprocess.check_output(
                ["git", "describe", "--always"]).strip().decode()
            self.git_branch = subprocess.check_output(
                "git rev-parse --abbrev-ref HEAD".split()).strip().decode()
            self.git_diff = subprocess.check_output(
                "git diff -- . :(exclude)Pipfile.lock".split()).strip().decode()[:49000]
        except subprocess.CalledProcessError:
            self.logger.info("Failed to get Git info...")

    @property
    def title(self):
        name = self.hyper_parameters.get("comment")
        if not name:
            name = f"Experiment on {self.corpus_name}"

        return f"{name} - {self.now_str} [#{randint(0, 999)}]"

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish(error=exc_type is not None)

    @abstractmethod
    def report_loss(self, loss_value: float, epoch: int):
        raise NotImplementedError()

    @abstractmethod
    def report_epoch(self, epoch: int, epoch_loss: float, train_las: float,
                     train_uas: float, dev_las: float, dev_uas: float):
        raise NotImplementedError()

    @abstractmethod
    def finish(self, error=False):
        raise NotImplementedError()


class GoogleSheetsReporter(BaseReporter):

    def __init__(self, sheets_id: str, auth_file_path: str, hyper_parameters: dict):
        super().__init__(hyper_parameters)
        self.template_engine = GoogleSheetsTemplateEngine(
            self.title, sheets_id, auth_file_path)

        # Init static values
        self.template_engine.update_variables(**self.hyper_parameters)
        self.template_engine.update_variables(
            title=self.title,
            status="Running",
            start_timestamp=self.start_time,
            corpus_name=self.corpus_name,
            params_json=json.dumps(self.hyper_parameters),
            git_diff=self.git_diff,
            git_commit=self.git_commit,
            git_branch=self.git_branch
        )

        # Iterate over the hyper parameters and add them to the table in multiple lines
        step = int(len(self.hyper_parameters) / 4)

        for i, item in enumerate(
                sorted(self.hyper_parameters.items(), key=lambda x: x[0])):
            param_num = min(int(i / step) + 1, 4)
            self.template_engine.update_variables(**{
                f"params_{param_num}_key": item[0],
                f"params_{param_num}_value": item[1]
            })

        # Save the changes in Google Docs
        self.template_engine.sync()

    def report_loss(self, loss_value: float, epoch: int):
        self.template_engine.worksheet.add_rows(1)
        self.template_engine.update_variables(
            loss_epoch=epoch,
            loss=loss_value,
            duration=self.duration
        )
        self.template_engine.sync()

    def report_epoch(self, epoch: int, epoch_loss: float, train_las: float,
                     train_uas: float, dev_las: float, dev_uas: float):
        self.template_engine.worksheet.add_rows(1)

        if dev_las > self.best_result["value"]:
            self.best_result = {
                "epoch": epoch, "value": dev_las
            }

        self.template_engine.update_variables(
            epoch=epoch,
            epoch_duration=self.now - self.last_epoch_start,
            epoch_loss=epoch_loss,
            epoch_training_las=train_las,
            epoch_training_uas=train_uas,
            epoch_develop_las=dev_las,
            epoch_develop_uas=dev_uas,
            best_epoch=self.best_result["epoch"],
            best_dev_las=self.best_result["value"],
            duration=self.duration
        )

        self.last_epoch_start = self.now
        self.template_engine.sync()

    def finish(self, error=False):
        self.template_engine.update_variables(
            status="Finished" if not error else "Error",
            duration=self.duration
        )
        self.template_engine.sync()


class DummyReporter(BaseReporter):

    def __init__(self, *args, **kwargs):
        super().__init__(hyper_parameters={})

    def report_loss(self, loss_value: float, epoch: int):
        pass

    def report_epoch(self, epoch: int, epoch_loss: float, train_las: float,
                     train_uas: float, dev_las: float, dev_uas: float):
        pass

    def finish(self, error=False):
        pass


def get_reporter(**kwargs):
    if kwargs.get("sheets_id") and kwargs.get("auth_file_path"):
        return GoogleSheetsReporter(**kwargs)
    else:
        return DummyReporter()
