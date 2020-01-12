from parseridge.parser.modules.data_parallel import Module


class Identity(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input, *args, **kwargs):
        return input
