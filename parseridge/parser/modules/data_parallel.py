from torch import nn

from parseridge.utils.logger import LoggerMixin


class DataParallel(nn.DataParallel):
    """
    Custom DataParallel class for multi-GPU training that allows accessing class
    attributes. See the tutorial for more information:
    https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
    """

    def __getattr__(self, name):
        return getattr(self.module, name)


class Module(nn.Module, LoggerMixin):
    def __init__(self, device="cpu", **kwargs):
        super(Module, self).__init__(**kwargs)
        self.device = device

    def parallel(self):
        return DataParallel(self)
