import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

        self.lstm_in_dim = self.embedding_dim
        self.lstm_out_dim = self.hidden_dim * 2  # x 2 because of BiRNN

        self.num_transitions = 4  # LEFT_ARC, RIGHT_ARC, SHIFT and SWAP
        self.num_labels = self.relations.num_relations

        # Take the top n entries from buffer / stack as input into the MLP
        self.stack_size = num_stack
        self.buffer_size = num_buffer

        self.word_embeddings = nn.Embedding(
            num_embeddings=len(self.vocabulary),
            embedding_dim=self.embedding_dim,
            padding_idx=self.vocabulary.get_id("<<<PADDING>>>")
        )

        self.lstm = nn.LSTM(
            input_size=self.lstm_in_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            dropout=dropout,
            bidirectional=True
        )

        self.dropout = nn.Dropout(p=dropout)

        self.transition_mlp = nn.Linear(
            (self.stack_size + self.buffer_size) * self.lstm_out_dim,
            self.num_transitions
        )

        self.relation_mlp = nn.Linear(
            (self.stack_size + self.buffer_size) * self.lstm_out_dim,
            self.num_labels
        )

        self._init_weights_xavier(self.word_embeddings)
        self._init_weights_xavier(self.lstm)
        self._init_weights_xavier(self.transition_mlp)
        self._init_weights_xavier(self.relation_mlp)

        # Declare tensors that are needed throughout the process
        self._mlp_padding = torch.zeros(
            self.lstm_out_dim, requires_grad=False).to(self.device)
        self.negative_infinity = torch.tensor(
            float("-inf"), requires_grad=False).to(self.device)
        self.one = torch.tensor(1.0, requires_grad=False).to(self.device)

    @staticmethod
    def _init_weights_xavier(network):
        """
        Initializes the layers of a given network with random values.
        Bias layers will be filled with zeros.

        Parameters
        ----------
        network : torch.nn object
            The network to initialize.
        """
        for name, param in network.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(
                    param, gain=nn.init.calculate_gain("tanh")
                )

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
        """

        Parameters
        ----------
        sentences
        sentence_features

        Returns
        -------

        """
        # Returns the LSTM outputs for every token in the sentence
        tokens = sentence_features[:, 0, :]
        tokens_embedded = self.word_embeddings(tokens)
        sentence_lengths = [len(sentence) for sentence in sentences]

        tokens_packed = pack_padded_sequence(
            tokens_embedded,
            torch.tensor(sentence_lengths, dtype=torch.int64),
            batch_first=True
        )

        packed_output, _ = self.lstm(tokens_packed)

        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True
        )

        return output

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
