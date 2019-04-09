import logging


class LoggerMixin:
    @property
    def logger(self):
        logger = logging.getLogger(type(self).__name__)
        if not logger.handlers:
            return self._logger_setup(type(self).__name__)

        return logger

    @staticmethod
    def _logger_setup(name):
        # create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-15s | %(levelname)-5s : %(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

        return logger
