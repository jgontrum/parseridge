import torch.nn as nn

from parseridge.parser.modules.data_parallel import Module


class MultilayerPerceptron(Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        dropout=0.0,
        activation=nn.Tanh,
        device="cpu",
    ):
        super().__init__(device=device)

        self.input_size = input_size
        self.output_size = output_size

        modules = []
        last_output_size = input_size
        for hidden_size in hidden_sizes:
            modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(last_output_size, hidden_size))
            modules.append(activation())
            last_output_size = hidden_size

        modules.append(nn.Linear(last_output_size, output_size))

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)

    @property
    def last_layer(self):
        return self.layers._modules[str(len(self.layers._modules) - 1)]
