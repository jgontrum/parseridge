import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from parseridge.analysis.param_change import ParameterChangeAnalyzer
from parseridge.parser.modules.actions_encoder import ActionsEncoder
from parseridge.parser.modules.attention import Attention
from parseridge.parser.modules.decoder import Decoder
from parseridge.parser.modules.input_encoder import InputEncoder
from parseridge.parser.modules.mlp import MultilayerPerceptron
from parseridge.parser.modules.utils import initialize_xavier_dynet_
from parseridge.utils.helpers import get_parameters
from parseridge.utils.logger import LoggerMixin


class ParseridgeModel(nn.Module, LoggerMixin):

    def __init__(self, relations, vocabulary,
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

        super(ParseridgeModel, self).__init__()
        """ Parameter definitions """
        self.device = device

        self.relations = relations
        self.vocabulary = vocabulary

        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size

        self.lstm_dropout = lstm_dropout
        self.mlp_dropout = mlp_dropout

        self.lstm_in_size = self.embedding_size
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

        """ Module definitions """

        """RNN that encodes the input sentence."""
        self.input_encoder = InputEncoder(
            token_vocabulary=vocabulary,
            token_embedding_size=embedding_size,
            hidden_size=lstm_hidden_size,
            layers=lstm_layers,
            dropout=lstm_dropout,
            device=device
        )

        """RNN that encodes a sequence of actions (e.g. transitions)."""
        self.actions_encoder = ActionsEncoder(
            input_size=32,
            output_size=64,
            num_layers=1
        )

        """ Computes attention over the output of the input encoder given the state of the
        action encoder. """
        self.attention = Attention(
            method="concat",
            hidden_size=self.input_encoder.output_size
        )

        """Given the output of the attention, computes a representation of the current
        configuration. Output is passed to the MLPs"""
        self.decoder = Decoder(
            input_size=self.input_encoder.output_size + self.actions_encoder.output_size,
            output_size=256
        )

        self.transition_mlp = MultilayerPerceptron(
            input_size=self.decoder.output_size,
            hidden_sizes=transition_mlp_layers,
            output_size=self.num_transitions,
            dropout=self.mlp_dropout,
            activation=nn.Tanh
        )

        self.relation_mlp = MultilayerPerceptron(
            input_size=self.decoder.output_size,
            hidden_sizes=relation_mlp_layers,
            output_size=self.num_labels,
            dropout=self.mlp_dropout,
            activation=nn.Tanh
        )

        initialize_xavier_dynet_(self)
        self.logger.info(f"Learning {len(get_parameters(self))} parameters.")

    # Hooks
    def before_batch(self):
        pass

    def after_batch(self):
        pass

    def before_epoch(self):
        self.zero_grad()

    def after_epoch(self):
        pass
        # self.param_analyzer.add_modules({
        #     "Word embeddings": self.word_embeddings,
        #     "LSTM": self.lstm,
        #     "Transition MLP": self.transition_mlp,
        #     "Relation MLP": self.relation_mlp,
        #     "Tra MLP Last": self.transition_mlp.last_layer,
        #     "Rel MLP Last": self.relation_mlp.last_layer,
        # }, report=True)

    def compute_lstm_output(self, sentences, sentence_features):
        outputs, _ =  self.input_encoder(sentences, sentence_features)
        return outputs

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

    @staticmethod
    def _concatenate_stack_and_buffer(stack, buffer):
        return torch.cat((stack, buffer), dim=1)

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

        mlp_input = self._concatenate_stack_and_buffer(stack_batch, buffer_batch)

        transitions_output = self.transition_mlp(mlp_input)
        relations_output = self.relation_mlp(mlp_input)

        return transitions_output, relations_output

    def forward(self, sentence_encoding_batch, action_encoding_batch, sentences,
                prev_decoder_hidden_state_batch, prev_decoder_cell_state_batch):

        batch_size = len(sentence_encoding_batch)

        # Turn the lists of tensors into one tensor with a batch dimension
        sentence_encoding_batch = torch.stack(sentence_encoding_batch)

        if prev_decoder_hidden_state_batch[0] is not None:
            prev_decoder_hidden_state_batch = torch.stack(prev_decoder_hidden_state_batch)
            prev_decoder_cell_state_batch = torch.stack(prev_decoder_cell_state_batch)
        else:
            prev_decoder_hidden_state_batch = None
            prev_decoder_cell_state_batch = None

        # Run attention over the input sentence using the latest action encoding as input
        attention_energies = self.attention(
            action_encoding_batch,
            sentence_encoding_batch,
            src_len=[len(s) for s in sentences]
        )

        context = attention_energies.bmm(sentence_encoding_batch)

        decoder_input = torch.cat((context, action_encoding_batch), dim=2)

        # Feed context vector into decoder
        mlp_input, decoder_hidden_states = self.decoder(
            decoder_input,
            prev_decoder_hidden_state_batch,
            prev_decoder_cell_state_batch,
            batch_size=batch_size
        )

        mlp_input = mlp_input.view(batch_size, -1)

        # Use output and feed it into MLP
        transitions_output = self.transition_mlp(mlp_input)
        relations_output = self.relation_mlp(mlp_input)

        return transitions_output, relations_output, decoder_hidden_states
