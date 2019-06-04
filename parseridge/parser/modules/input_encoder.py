import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.mlp import MultilayerPerceptron
from parseridge.parser.modules.positional_embeddings import PositionalEmbeddings


class InputEncoder(Module):
    def __init__(self, token_vocabulary, token_embedding_size,
                 hidden_size, layers=2, dropout=0.33, max_sentence_length=100,
                 positional_embedding_size=128, sum_directions=True,
                 reduce_dimensionality=False, **kwargs):
        super(InputEncoder, self).__init__(**kwargs)

        self.token_vocabulary = token_vocabulary
        self.input_size = token_embedding_size
        self.hidden_size = hidden_size
        self.positional_embedding_size = positional_embedding_size
        self.max_sentence_length = max_sentence_length
        self.sum_directions = sum_directions
        self.reduce_dimensionality = reduce_dimensionality

        self.output_size = hidden_size if self.sum_directions else 2 * hidden_size

        self.token_embeddings = nn.Embedding(
            num_embeddings=len(self.token_vocabulary),
            embedding_dim=token_embedding_size,
            padding_idx=self.token_vocabulary.get_id("<<<PADDING>>>"),
        )

        if self.positional_embedding_size:
            self.position_embeddings = PositionalEmbeddings(
                embedding_size=self.positional_embedding_size,
                max_length=self.max_sentence_length,
                device=self.device
            )

            self.output_size += self.positional_embedding_size

        # TODO Add other feature embeddings here
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        if self.reduce_dimensionality:
            self.mlp = MultilayerPerceptron(
                input_size=self.output_size,
                output_size=128,
                hidden_sizes=[256],
                dropout=0.33,
                activation=nn.Tanh
            )

            self.output_size = self.mlp.output_size

    def load_external_embeddings(self, embeddings):
        self.logger.info("Loading external embeddings into the embedding layer...")
        self.token_embeddings.weight = embeddings.get_weight_matrix(
            self.token_vocabulary, self.device)

    def forward(self, sentence_batch, sentence_lengths):
        tokens_embedded = self.token_embeddings(sentence_batch)

        input_packed = pack_padded_sequence(
            tokens_embedded,
            lengths=sentence_lengths,
            batch_first=True
        )

        packed_outputs, hidden = self.rnn(input_packed)

        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True
        )

        if self.sum_directions:
            outputs = (
                    outputs[:, :, :self.hidden_size] +
                    outputs[:, :, self.hidden_size:]
            )  # Sum bidirectional outputs

        if self.reduce_dimensionality:
            outputs = self.mlp(outputs)

        if self.positional_embedding_size:
            positional_embeddings = self.position_embeddings(sentence_lengths)
            outputs = torch.cat((outputs, positional_embeddings), dim=2)

        return outputs, hidden
