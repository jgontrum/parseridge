import torch

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from parseridge.parser.modules.utils import init_weights_xavier


class InputEncoder(nn.Module):
    def __init__(self, token_vocabulary, token_embedding_size,
                 hidden_size, layers=2, dropout=0.33):
        super(InputEncoder, self).__init__()

        self.token_vocabulary = token_vocabulary

        self.token_embeddings = nn.Embedding(
            num_embeddings=len(self.token_vocabulary),
            embedding_dim=token_embedding_size,
            padding_idx=self.token_vocabulary.get_id("<<<PADDING>>>")
        )
        init_weights_xavier(self.token_embeddings)

        # !! Add other feature embeddings here !!

        self.input_size = token_embedding_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        # self.output_size might change depending on how the output of the
        # BiRNN is handled (summed vs. concatenated)

        self.birnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout,
            bidirectional=True
        )
        init_weights_xavier(self.birnn)

    def _get_token_embeddings(self, sentence_features):
        tokens = sentence_features[:, 0, :]
        return self.token_embeddings(tokens)

    def forward(self, sentences, sentence_features):
        sentence_lengths = [len(sentence) for sentence in sentences]

        tokens_embedded = self._get_token_embeddings(sentence_features)
        # Add other embedding lookups here

        input_packed = pack_padded_sequence(
            tokens_embedded,  # Concatenate all embeddings / features here
            torch.tensor(sentence_lengths, dtype=torch.int64),
            batch_first=True
        )

        packed_outputs, hidden = self.birnn(input_packed)

        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True
        )

        outputs = (
                outputs[:, :, :self.hidden_size] +
                outputs[:, :, self.hidden_size:]
        )  # Sum bidirectional outputs

        return outputs, hidden
