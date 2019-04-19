import torch

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class InputEncoder(nn.Module):
    def __init__(self, token_vocabulary, token_embedding_size,
                 hidden_size, layers=2, dropout=0.33, device="cpu"):
        super(InputEncoder, self).__init__()

        self.token_vocabulary = token_vocabulary
        self.input_size = token_embedding_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.device = device

        self.token_embeddings = nn.Embedding(
            num_embeddings=len(self.token_vocabulary),
            embedding_dim=token_embedding_size,
            padding_idx=self.token_vocabulary.get_id("<<<PADDING>>>"),
        )

        # TODO Add other feature embeddings here
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

    def _get_token_embeddings(self, sentence_features):
        tokens = sentence_features[:, 0, :]
        return self.token_embeddings(tokens)

    def forward(self, sentences, sentence_features):
        sentence_lengths = [len(sentence) for sentence in sentences]

        tokens_embedded = self._get_token_embeddings(sentence_features)
        # TODO Add other embedding lookups here

        input_packed = pack_padded_sequence(
            tokens_embedded,
            torch.tensor(sentence_lengths, dtype=torch.int64, device=self.device),
            batch_first=True
        )

        packed_outputs, hidden = self.rnn(input_packed)

        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True
        )

        outputs = (
                outputs[:, :, :self.hidden_size] +
                outputs[:, :, self.hidden_size:]
        )  # Sum bidirectional outputs

        return outputs, hidden
