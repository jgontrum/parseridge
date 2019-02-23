import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from parseridge.parser.modules.attention import Attention
from parseridge.parser.modules.input_encoder import InputEncoder
from parseridge.parser.modules.mlp import MultilayerPerceptron
from parseridge.utils.helpers import Action
from parseridge.utils.helpers import Transition as T


class ParseridgeModel(nn.Module):

    def __init__(self, relations, vocabulary, dropout=0.33, embedding_dim=100,
                 hidden_dim=125, num_stack=3, num_buffer=1, device="cpu"):
        super(ParseridgeModel, self).__init__()
        self.device = device

        self.relations = relations
        self.vocabulary = vocabulary
        self.optimizer = None  # Should be set after creation.

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.num_transitions = 4  # LEFT_ARC, RIGHT_ARC, SHIFT and SWAP
        self.num_labels = self.relations.num_relations

        # Take the top n entries from buffer / stack as input into the MLP
        self.stack_size = num_stack
        self.buffer_size = num_buffer

        self.input_encoder = InputEncoder(
            vocabulary, embedding_dim, hidden_dim
        )
        output_size = self.input_encoder.output_size

        self.attention = Attention("general", output_size, self.device)

        self.dropout = nn.Dropout(p=dropout)

        self.transition_mlp = MultilayerPerceptron(
            (self.stack_size + self.buffer_size) * output_size,
            125,
            self.num_transitions
        )

        self.relation_mlp = MultilayerPerceptron(
            (self.stack_size + self.buffer_size) * output_size,
            125,
            self.num_labels
        )

        # Declare tensors that are needed throughout the process
        self._mlp_padding = torch.zeros(
            output_size, requires_grad=True).to(self.device)
        self.negative_infinity = torch.tensor(
            float("-inf"), requires_grad=True).to(self.device)
        self.one = torch.tensor(1.0, requires_grad=True).to(self.device)


    def _pad_list(self, list_, length):
        """
        Pad a given list with zeros so that it can be fed into the MLP.

        Parameters
        ----------
        list_ : List of tensor
            List to be padded
        length : int
            Desired length of the output

        Returns
        -------
        list_ : List of tensor
            Padded list
        """
        pad_size = length - len(list_)
        assert pad_size >= 0

        for _ in range(pad_size):
            list_.append(self._mlp_padding)

        return list_

    def compute_lstm_output(self, sentences, sentence_features):
        return self.input_encoder(sentences, sentence_features)

    def compute_mlp_output(self, lstm_out_batch, stack_index_batch,
                           buffer_index_batch):
        """

        :param lstm_out:
        :param stack_index:
        :param buffer_index:
        :return:
        """

        stack_batch = []
        for stack_index, lstm_out in zip(stack_index_batch, lstm_out_batch):
            stack = self._pad_list([
                lstm_out[i] for i in stack_index[-self.stack_size:]
            ], self.stack_size)

            stack = torch.stack(stack).view((-1,))
            stack.requires_grad_()
            stack_batch.append(stack)

        stack_batch = torch.stack(tuple(stack_batch))

        buffer_batch = []
        for buffer_index, lstm_out in zip(buffer_index_batch, lstm_out_batch):
            buffer = self._pad_list([
                lstm_out[i] for i in buffer_index[:self.buffer_size]
            ], self.buffer_size)

            buffer = torch.stack(buffer).view((-1,))
            buffer.requires_grad_()
            buffer_batch.append(buffer)

        buffer_batch = torch.stack(tuple(buffer_batch))

        mlp_input = torch.cat((stack_batch, buffer_batch), dim=1)
        mlp_input = self.dropout(mlp_input)

        transitions_output = torch.tanh(
            self.transition_mlp(mlp_input)
        )

        relations_output = torch.tanh(
            self.relation_mlp(mlp_input)
        )

        return transitions_output, relations_output


    def perform_back_propagation(self, loss):
        """

        Parameters
        ----------
        loss

        Returns
        -------

        """
        metric = 0
        if len(loss) >= 50:
            batch_loss = sum(loss)
            batch_loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            metric = batch_loss.item() + len(loss)
            loss = []

        return loss, metric

    def forward(self, *input):
        """
        Implemented because the class should implement all abstract methods
        of nn.Model.
        """
        return self.compute_mlp_output(*input)
