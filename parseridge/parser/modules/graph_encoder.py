import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from parseridge.parser.modules.utils import init_weights_xavier


class GraphEncoder(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.33, device="cpu"):
        super(GraphEncoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.device = device

        self.graph_encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.output_size,
            num_layers=2,
            dropout=dropout,
            bidirectional=True
        )
        init_weights_xavier(self.graph_encoder)

    def forward(self, sentences):
        sentence_lengths = [len(sentence) for sentence in sentences]

        graphs = torch.tensor([
            sentence.get_graph_matrix()
            for sentence in sentences
        ]).to(self.device)

        input_packed = pack_padded_sequence(
            graphs,
            torch.tensor(sentence_lengths, dtype=torch.int8),
            batch_first=True
        )

        packed_outputs, hidden = self.graph_encoder(input_packed)

        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True
        )

        outputs = (
                outputs[:, :, :self.hidden_size] +
                outputs[:, :, self.hidden_size:]
        )  # Sum bidirectional outputs

        return outputs, hidden
