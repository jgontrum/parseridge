import numpy as np

from parseridge.utils.helpers import get_parameters
from parseridge.utils.logger import LoggerMixin


class ParameterChangeAnalyzer(LoggerMixin):
    def __init__(self):
        self.last_parameters = None

    def add_modules(self, modules, report=False):
        new_parameters = {
            name: np.abs(get_parameters(params)) for name, params in modules.items()
        }

        if self.last_parameters is None:
            self.last_parameters = new_parameters
            return

        # Compute changes
        diffs = {
            name: abs(1 - (np.mean(last_param) / np.mean(new_parameters[name])))
            for name, last_param in self.last_parameters.items()
        }

        self.last_parameters = new_parameters

        if report:
            log = ["Parameter changes:"]
            for name, diff in sorted(diffs.items(), key=lambda x: x[0]):
                log.append(f"{name:<15}: {diff * 100:2.8f}%")
            self.logger.info("\n".join(log))

        return diffs
