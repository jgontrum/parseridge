import logging


class LoggerMixin:
    @property
    def logger(self):
        return logging.getLogger(f"parseridge.{type(self).__name__}")
