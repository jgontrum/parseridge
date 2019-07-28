from torch import nn, Tensor

from parseridge.parser.modules.data_parallel import Module


class AddAndNormLayer(Module):
    def __init__(self, model_size: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.input_size = self.output_size = model_size

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(model_size)

    def forward(self, input: Tensor, output: Tensor):
        return self.layer_norm(input + self.dropout(output))
