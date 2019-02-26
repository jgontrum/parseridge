import torch.nn as nn

from parseridge.parser.modules.utils import init_weights_xavier


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size,
                 activation="tanh"):
        super(MultilayerPerceptron, self).__init__()

        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU
        }

        modules = []
        last_output_size = input_size
        for hidden_size in hidden_sizes:
            modules.append(nn.Linear(last_output_size, hidden_size))
            modules.append(activations[activation]())
            last_output_size = hidden_size

        modules.append(nn.Linear(last_output_size, output_size))

        self.layers = nn.Sequential(*modules)

        init_weights_xavier(self.layers, activation)

    def forward(self, x):
        return self.layers(x)
