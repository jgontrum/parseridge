import torch
from torch import nn

from parseridge.parser.modules.attention import Attention
from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.positional_embeddings import PositionalEmbeddings
from parseridge.parser.modules.rnn import RNN
from parseridge.parser.modules.utils import create_mask, lookup_tensors_for_indices, \
    initialize_xavier_dynet_


class SequenceAttention(Module):
    """Wrapper to generate a representation for a stack/buffer sequence."""

    def __init__(self, input_size, lstm_size, positional_embedding_size, **kwargs):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.lstm_size = lstm_size
        self.positional_embedding_size = positional_embedding_size

        self.padding_tensor_param = nn.Parameter(
            torch.zeros(self.input_size, dtype=torch.float)
        )

        if self.lstm_size:
            self.rnn = RNN(
                rnn=nn.LSTM(
                    input_size=self.input_size,
                    hidden_size=self.lstm_size,
                    bidirectional=True,
                    batch_first=True
                ),
                device=self.device
            )

        attention_input_size = self.lstm_size * 2 if self.lstm_size else self.input_size

        if self.positional_embedding_size:
            self.positional_embeddings = PositionalEmbeddings(
                embedding_size=self.positional_embedding_size,
                max_length=80,
                device=self.device
            )

            attention_input_size += self.positional_embeddings.output_size

        self.attention_layer = Attention(
            input_size=attention_input_size,
            device=self.device
        )

        initialize_xavier_dynet_(self.padding_tensor_param)
        self.output_size = self.attention_layer.output_size

    @property
    def padding_tensor(self):
        return torch.tanh(self.padding_tensor_param)

    def forward(self, indices_batch, sentence_encoding_batch):
        batch = lookup_tensors_for_indices(
            indices_batch=indices_batch,
            sequence_batch=sentence_encoding_batch,
            padding=self.padding_tensor,
            size=max([len(s) for s in indices_batch])
        )

        batch_mask = create_mask(
            [len(sequence) for sequence in indices_batch],
            max_len=batch.size(1),
            device=self.device
        )

        if self.lstm_size:
            batch = self.rnn(
                input=batch,
                sequences=indices_batch
            )

        if self.positional_embedding_size:
            positional_embeddings = self.positional_embeddings(
                [len(sequence) for sequence in indices_batch])

            batch = torch.cat((batch, positional_embeddings), dim=2)

        batch_attn = self.attention_layer(
            batch, mask=batch_mask
        )

        return batch_attn
