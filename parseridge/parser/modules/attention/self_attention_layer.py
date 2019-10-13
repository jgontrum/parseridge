from typing import Tuple

from torch import nn, Tensor
from torch.nn import MultiheadAttention

from parseridge.parser.modules.add_and_norm_layer import AddAndNormLayer
from parseridge.parser.modules.data_parallel import Module


class SelfAttentionLayer(Module):
    def __init__(self, model_size: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.input_size = self.output_size = model_size
        self.num_heads = num_heads

        self.multihead_attention = MultiheadAttention(
            embed_dim=self.input_size, num_heads=num_heads
        )

        self.attention_norm = AddAndNormLayer(model_size=self.input_size)

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=4 * self.input_size),
            nn.ReLU(),
            nn.Linear(in_features=4 * self.input_size, out_features=self.input_size),
        )

        self.linear_layer_norm = AddAndNormLayer(model_size=self.input_size)

    def forward(self, sequence: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        # [Batch, Sequence, Embedding] -> [Sequence, Batch, Embedding]
        sequence_by_sent = sequence.transpose(0, 1)

        attention_output, attention_weights = self.multihead_attention(
            query=sequence_by_sent,
            key=sequence_by_sent,
            value=sequence_by_sent,
            key_padding_mask=mask,
        )

        # [Sequence, Batch, Embedding] -> [Batch, Sequence, Embedding]
        attention_output = attention_output.transpose(0, 1)

        attention_output = self.attention_norm(input=sequence, output=attention_output)

        attention_output = self.linear_layer_norm(
            input=attention_output, output=self.linear_layer(attention_output)
        )

        return attention_output, attention_weights
