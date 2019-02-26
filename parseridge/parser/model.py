import numpy as np

import torch
import torch.nn as nn

from parseridge.parser.modules.attention import Attention
from parseridge.parser.modules.graph_encoder import GraphEncoder
from parseridge.parser.modules.input_encoder import InputEncoder
from parseridge.parser.modules.mlp import MultilayerPerceptron


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

        self.max_sentence_size = 100

        self.input_encoder = InputEncoder(
            vocabulary, embedding_dim, hidden_dim
        )
        output_size = self.input_encoder.output_size
        self.attention = Attention("concat", output_size, self.device)

        # self.graph_encoder = GraphEncoder(
        #     input_size=self.max_sentence_size,
        #     output_size=32,
        #     device=self.device
        # )

        self.dropout = nn.Dropout(p=dropout)

        self.transition_mlp = MultilayerPerceptron(
            output_size + self.max_sentence_size ** 2,
            [125],
            self.num_transitions
        )

        self.relation_mlp = MultilayerPerceptron(
            output_size + self.max_sentence_size ** 2,
            [125],
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
        encoder_outputs, hidden_states = \
            self.input_encoder(sentences, sentence_features)

        attention_weights = self.attention(
            hidden_states, encoder_outputs)

        context_vectors = attention_weights.bmm(encoder_outputs)
        context_vectors = context_vectors.squeeze(1)
        return encoder_outputs, context_vectors

    def compute_mlp_output(self, context_vector_batch, sentences):
        """

        :param lstm_out:
        :param stack_index:
        :param buffer_index:
        :return:
        """
        context_vector_batch = torch.stack(context_vector_batch)

        graphs_batch = []
        for sentence in sentences:
            graph = sentence.get_graph_matrix()
            padding_size = self.max_sentence_size - graph.shape[0]
            graph = np.pad(
                graph, (0, padding_size), 'constant', constant_values=0
            )
            graphs_batch.append(graph.flatten())

        graphs_batch = torch.tensor(
            graphs_batch, dtype=torch.float).to(self.device)

        mlp_input = torch.cat(
            (graphs_batch, context_vector_batch), dim=1)

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
