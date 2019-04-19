import numpy as np

import torch
import torch.nn as nn


class ActionsEncoder(nn.Module):
    def __init__(self, input_size, output_size, num_layers, device="cpu"):
        super(ActionsEncoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.transition_embeddings = nn.Embedding(
            num_embeddings=4,
            embedding_dim=input_size
        )

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=self.output_size,
            num_layers=self.num_layers,
            bidirectional=False,
            batch_first=True
        )

        self.initial_output = torch.zeros(
            self.output_size, requires_grad=True).to(self.device)

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

    def get_initial_state(self, batch_size):
        # Initial state, no transition has been made so far. Return a zero filled
        # tensor and initialize the hidden state / cell state tensors.
        return (torch.zeros((batch_size, 1, self.output_size), device=self.device),
                self.init_hidden(batch_size))

    def forward(self, action_batch, prev_hidden_state_batch, prev_cell_state_batch):
        prev_hidden_state_batch = torch.stack(prev_hidden_state_batch)
        prev_hidden_state_batch = prev_hidden_state_batch.transpose(0, 1)

        prev_cell_state_batch = torch.stack(prev_cell_state_batch)
        prev_cell_state_batch = prev_cell_state_batch.transpose(0, 1)

        transitions = np.array([action.transition.value for action in action_batch])
        if len(transitions.shape) == 1:
            transitions = np.array([transitions])

        transitions = torch.tensor(transitions, dtype=torch.long, device=self.device)
        transitions = transitions.view(-1, 1)

        transitions_emb = self.transition_embeddings(transitions)

        # Turn it into a batch of inputs of length 1

        output, (hidden_state, cell_state) = self.rnn(
            transitions_emb, (prev_hidden_state_batch, prev_cell_state_batch))

        return output, (hidden_state, cell_state)
