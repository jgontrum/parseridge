import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, device="cpu"):
        super(Decoder, self).__init__()
        self.device = device
        self.output_size = output_size
        self.num_layers = 1

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=self.output_size,
            num_layers=self.num_layers,
            bidirectional=False,
            batch_first=True
        )

    def init_hidden(self, batch_size):
        # See https://discuss.pytorch.org/t/correct-way-to-declare-hidden-and-cell-states-of-lstm/15745/2
        template_tensor = next(self.parameters()).data
        hidden_state = torch.zeros(
            self.num_layers, batch_size, self.output_size, device=self.device,
            requires_grad=True
        )
        cell_state = torch.zeros(
            self.num_layers, batch_size, self.output_size, device=self.device,
            requires_grad=True
        )

        return hidden_state, cell_state

    def forward(self, context_vector_batch, prev_hidden_state_batch,
                prev_cell_state_batch, batch_size=None):

        if context_vector_batch is None or prev_hidden_state_batch is None:
            prev_hidden_state_batch, prev_cell_state_batch = self.init_hidden(batch_size)
        else:
            prev_hidden_state_batch = prev_hidden_state_batch.transpose(0, 1)
            prev_cell_state_batch = prev_cell_state_batch.transpose(0, 1)

        output, (hidden_state, cell_state) = self.rnn(
            context_vector_batch, (prev_hidden_state_batch, prev_cell_state_batch))

        return output, (hidden_state, cell_state)
