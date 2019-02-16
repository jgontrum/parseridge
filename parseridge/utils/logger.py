import logging


class LoggerMixin:
    @property
    def logger(self):
        return logging.getLogger(type(self).__name__)