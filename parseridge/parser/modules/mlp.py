import torch.nn as nn

from parseridge.parser.modules.utils import init_weights_xavier


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultilayerPerceptron, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

        init_weights_xavier(self.layers)

    def forward(self, x):
        return self.layers(x)
