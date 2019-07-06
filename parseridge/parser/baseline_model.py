import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from parseridge.corpus.vocabulary import Vocabulary
from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.mlp import MultilayerPerceptron
from parseridge.parser.modules.utils import initialize_xavier_dynet_


class BaselineModel(Module):

    def __init__(self, relations, vocabulary: Vocabulary,
                 num_stack=3,
                 num_buffer=1,
                 lstm_dropout=0.33,
                 mlp_dropout=0.25,
                 embedding_size=100,
                 lstm_hidden_size=125,
                 lstm_layers=2,
                 transition_mlp_layers=None,
                 relation_mlp_layers=None,
                 device="cpu"):

        super().__init__(device=device)

        """ Parameter definitions """
        self.relations = relations
        self.vocabulary = vocabulary

        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size

        self.lstm_dropout = lstm_dropout
        self.mlp_dropout = mlp_dropout

        self.lstm_in_size = self.embedding_size
        self.lstm_out_size = self.lstm_hidden_size * 2  # x 2 because of BiRNN
        self.lstm_layers = lstm_layers

        if relation_mlp_layers is None:
            relation_mlp_layers = [100]

        if transition_mlp_layers is None:
            transition_mlp_layers = [100]

        self.num_transitions = 4  # LEFT_ARC, RIGHT_ARC, SHIFT and SWAP
        self.num_labels = len(self.relations)

        # Take the top n entries from buffer / stack as input into the MLP
        self.stack_size = num_stack
        self.buffer_size = num_buffer

        self.mlp_in_size = (self.stack_size + self.buffer_size) * self.lstm_out_size

        """ Module definitions """

        self.word_embeddings = nn.Embedding(
            num_embeddings=len(self.vocabulary),
            embedding_dim=self.embedding_size,
            padding_idx=self.vocabulary.get_id("<<<PADDING>>>"),
        )

        self.lstm = nn.LSTM(
            input_size=self.lstm_in_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.lstm_dropout,
            bidirectional=True,
            batch_first=True
        )

        self.transition_mlp = MultilayerPerceptron(
            input_size=self.mlp_in_size,
            hidden_sizes=transition_mlp_layers,
            output_size=self.num_transitions,
            dropout=self.mlp_dropout,
            activation=nn.Tanh
        )

        self.relation_mlp = MultilayerPerceptron(
            input_size=self.mlp_in_size,
            hidden_sizes=relation_mlp_layers,
            output_size=self.num_labels,
            dropout=self.mlp_dropout,
            activation=nn.Tanh
        )

        self._mlp_padding_param = nn.Parameter(
            torch.zeros(self.lstm_out_size, dtype=torch.float)
        )

        initialize_xavier_dynet_(self)

    def reset_(self) -> None:
        self._mlp_padding = nn.Tanh()(self._mlp_padding_param)

    def get_contextualized_input(self, sentences, sentence_features):
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
            torch.tensor(sentence_lengths, dtype=torch.int64, device=self.device),
            batch_first=True
        )

        packed_output, _ = self.lstm(tokens_packed)

        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True
        )

        return output.contiguous()

    @staticmethod
    def _get_tensor_for_indices(indices_batch, lstm_out_batch, padding, size):
        batch = []
        for index_list, lstm_out in zip(indices_batch, lstm_out_batch):
            items = [lstm_out[i] for i in index_list]
            while len(items) < size:
                items.append(padding)

            # Turn list of tensors into one tensor
            items = torch.stack(items)

            # Flatten the tensor
            items = items.view((-1,)).contiguous()

            batch.append(items)

        return torch.stack(batch).contiguous()

    def compute_mlp_output(self, lstm_out_batch, stack_index_batch, buffer_index_batch):
        stack_batch = self._get_tensor_for_indices(
            indices_batch=[stack[-self.stack_size:] for stack in stack_index_batch],
            lstm_out_batch=lstm_out_batch,
            padding=self._mlp_padding,
            size=self.stack_size
        )

        buffer_batch = self._get_tensor_for_indices(
            indices_batch=[buffer[:self.buffer_size] for buffer in buffer_index_batch],
            lstm_out_batch=lstm_out_batch,
            padding=self._mlp_padding,
            size=self.buffer_size
        )

        mlp_input = torch.cat((stack_batch, buffer_batch), dim=1)

        transitions_output = self.transition_mlp(mlp_input)
        relations_output = self.relation_mlp(mlp_input)

        return transitions_output, relations_output
