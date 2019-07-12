import logging


class LoggerMixin:
    @property
    def logger(self):
        return logging.getLogger(f"main.{type(self).__name__}")
