import logging


class LoggerMixin:
    @property
    def logger(self):
        return logging.getLogger(f"{type(self).__name__}")
