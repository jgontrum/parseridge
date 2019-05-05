import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from parseridge.analysis.param_change import ParameterChangeAnalyzer
from parseridge.parser.modules.actions_encoder import ActionsEncoder
from parseridge.parser.modules.attention import Attention
from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.decoder import Decoder
from parseridge.parser.modules.dependency_graph_encoder import DependencyGraphEncoder
from parseridge.parser.modules.input_encoder import InputEncoder
from parseridge.parser.modules.mlp import MultilayerPerceptron
from parseridge.parser.modules.rnn import RNN
from parseridge.parser.modules.utils import initialize_xavier_dynet_, create_mask
from parseridge.utils.helpers import get_parameters
from parseridge.utils.logger import LoggerMixin


class ParseridgeModel(Module):

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

        self.embedding_size = 300
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

        """Computes attention over the output of the input encoder given the state of the
        action encoder. """
        self.stack_attention = Attention(
            input_size=self.input_encoder.output_size * 2,
            device=device
        )

        self.stack_rnn = RNN(
            rnn=nn.LSTM(
                input_size=self.input_encoder.output_size,
                hidden_size=self.input_encoder.output_size,
                bidirectional=True,
                batch_first=True
            ),
            device=self.device
        )

        """Computes attention over the output of the input encoder given the state of the
        action encoder. """
        self.buffer_attention = Attention(
            input_size=self.input_encoder.output_size * 2,
            device=device
        )

        self.buffer_rnn = RNN(
            rnn=nn.LSTM(
                input_size=self.input_encoder.output_size,
                hidden_size=self.input_encoder.output_size,
                bidirectional=True,
                batch_first=True
            ),
            device=self.device
        )

        self.transition_mlp = MultilayerPerceptron(
            input_size=self.input_encoder.output_size * 4,
            hidden_sizes=transition_mlp_layers,
            output_size=self.num_transitions,
            dropout=self.mlp_dropout,
            activation=nn.Tanh,
            device=device
        )

        self.relation_mlp = MultilayerPerceptron(
            input_size=self.input_encoder.output_size * 4,
            hidden_sizes=relation_mlp_layers,
            output_size=self.num_labels,
            dropout=self.mlp_dropout,
            activation=nn.Tanh,
            device=device
        )

        self._mlp_padding_param = nn.Parameter(
            torch.zeros(self.input_encoder.output_size, dtype=torch.float)
        )

        initialize_xavier_dynet_(self)

        self.logger.info("Loading external word embeddings...")
        # self.input_encoder.load_external_embeddings()
        self.logger.info(f"Learning {len(get_parameters(self))} parameters.")

    # Hooks
    def before_batch(self):
        self._mlp_padding = torch.tanh(self._mlp_padding_param)

    def after_batch(self):
        pass

    def before_epoch(self):
        self.zero_grad()

    def after_epoch(self):
        pass

    def compute_lstm_output(self, sentences, sentence_features):
        outputs, _ =  self.input_encoder(sentences, sentence_features)
        return outputs

    @staticmethod
    def _get_tensor_for_indices(indices_batch, lstm_out_batch, padding, size):
        size = max(1, size)
        batch = []
        for index_list, lstm_out in zip(indices_batch, lstm_out_batch):
            items = [lstm_out[i] for i in index_list]
            while len(items) < size:
                items.append(padding)

            # Turn list of tensors into one tensor
            items = torch.stack(items)

            # Flatten the tensor
            # items = items.view((-1,)).contiguous()

            batch.append(items)

        return torch.stack(batch).contiguous()

    @staticmethod
    def _concatenate_stack_and_buffer(stack, buffer):
        return torch.cat((stack, buffer), dim=1)

    def compute_legacy_mlp_input(self, lstm_out_batch, stack_index_batch, buffer_index_batch):
        stack_batch = self._get_tensor_for_indices(
            indices_batch=[stack for stack in stack_index_batch],
            lstm_out_batch=lstm_out_batch,
            padding=self._mlp_padding,
            size=max([len(s) for s in stack_index_batch])
        )

        stack_batch = torch.sum(stack_batch, dim=1)

        buffer_batch = self._get_tensor_for_indices(
            indices_batch=[buffer for buffer in buffer_index_batch],
            lstm_out_batch=lstm_out_batch,
            padding=self._mlp_padding,
            size=max([len(s) for s in buffer_index_batch])
        )

        buffer_batch = torch.sum(buffer_batch, dim=1)

        return self._concatenate_stack_and_buffer(stack_batch, buffer_batch)

    def forward(self, sentence_encoding_batch, action_encoding_batch, sentences,
                predicted_sentences_batch,
                # prev_decoder_hidden_state_batch, prev_decoder_cell_state_batch,
                stack_index_batch=None, buffer_index_batch=None, use_legacy=False,
                ):

        # Turn the lists of tensors into one tensor with a batch dimension
        sentence_encoding_batch = torch.stack(sentence_encoding_batch)

        stack_batch = self._get_tensor_for_indices(
            indices_batch=stack_index_batch,
            lstm_out_batch=sentence_encoding_batch,
            padding=self._mlp_padding,
            size=max([len(s) for s in stack_index_batch])
        )

        stack_batch = self.stack_rnn(
            input=stack_batch,
            sequences=stack_index_batch,
            ignore_empty_sequences=False
        )

        stack_batch_mask = create_mask(
            [len(s) for s in stack_index_batch],
            max_len=stack_batch.size(1),
            device=self.device
        )

        stack_batch_attn = self.stack_attention(
            stack_batch, mask=stack_batch_mask
        )

        buffer_batch = self._get_tensor_for_indices(
            indices_batch=buffer_index_batch,
            lstm_out_batch=sentence_encoding_batch,
            padding=self._mlp_padding,
            size=max([len(s) for s in buffer_index_batch])
        )

        buffer_batch_mask = create_mask(
            [len(s) for s in buffer_index_batch],
            max_len=buffer_batch.size(1),
            device=self.device
        )

        buffer_batch = self.buffer_rnn(
            input=buffer_batch,
            sequences=buffer_index_batch,
            ignore_empty_sequences=False
        )

        buffer_batch_attn = self.stack_attention(
            buffer_batch, mask=buffer_batch_mask
        )

        mlp_input = torch.cat((stack_batch_attn, buffer_batch_attn), dim=1)

        # Use output and feed it into MLP
        transitions_output = self.transition_mlp(mlp_input)
        relations_output = self.relation_mlp(mlp_input)

        return transitions_output, relations_output#, decoder_hidden_states
