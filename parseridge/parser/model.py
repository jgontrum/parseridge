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

        """RNN that encodes a sequence of actions (e.g. transitions)."""
        self.actions_encoder = ActionsEncoder(
            input_size=32,
            output_size=64,
            num_layers=1,
            device=device
        )

        """Computes attention over the output of the input encoder given the state of the
        action encoder. """
        self.attention = Attention(
            input_size=self.input_encoder.output_size,
            device=device
        )

        """RNN to encode the partial dependency graph."""
        self.dependency_graph_encoder = DependencyGraphEncoder(
            input_size=lstm_hidden_size,
            output_size=512,
            relations=self.relations,
            device=device
        )

        """Given the output of the attention, computes a representation of the current
        configuration. Output is passed to the MLPs"""
        self.decoder = Decoder(
            input_size=1149, #self.input_encoder.output_size + self.actions_encoder.output_size,
            output_size=512,
            device=device
        )


        self.transition_mlp = MultilayerPerceptron(
            input_size=125, #self.input_encoder.output_size * 4 + 1024,
            hidden_sizes=transition_mlp_layers,
            output_size=self.num_transitions,
            dropout=self.mlp_dropout,
            activation=nn.Tanh,
            device=device
        )

        self.relation_mlp = MultilayerPerceptron(
            input_size=125,
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

    def compute_legacy_mlp_input(self, lstm_out_batch, stack_index_batch, buffer_index_batch):
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

        return self._concatenate_stack_and_buffer(stack_batch, buffer_batch)

    def forward(self, sentence_encoding_batch, action_encoding_batch, sentences,
                predicted_sentences_batch,
                # prev_decoder_hidden_state_batch, prev_decoder_cell_state_batch,
                stack_index_batch=None, buffer_index_batch=None, use_legacy=False,
                ):

        batch_size = len(sentence_encoding_batch)

        if use_legacy and stack_index_batch and buffer_index_batch:
            mlp_input = self.compute_legacy_mlp_input(
                sentence_encoding_batch, stack_index_batch, buffer_index_batch
            )

            graph_encoding = self.dependency_graph_encoder(
                predicted_sentences_batch, sentence_encoding_batch)

            mlp_input = torch.cat((mlp_input, graph_encoding), dim=1)

        else:
            # Turn the lists of tensors into one tensor with a batch dimension
            sentence_encoding_batch = torch.stack(sentence_encoding_batch)

            # if prev_decoder_hidden_state_batch[0] is not None:
            #     prev_decoder_hidden_state_batch = torch.stack(
            #         prev_decoder_hidden_state_batch)
            #     prev_decoder_cell_state_batch = torch.stack(prev_decoder_cell_state_batch)
            # else:
            #     prev_decoder_hidden_state_batch = None
            #     prev_decoder_cell_state_batch = None
            #
            # graph_encoding = self.dependency_graph_encoder(
            #     predicted_sentences_batch, sentence_encoding_batch).unsqueeze(1)

            # Run attention over the input sentence using the latest action encoding as input
            # attention_energies = self.attention(
            #     graph_encoding,
            #     sentence_encoding_batch,
            #     src_len=[len(s) for s in sentences]
            # )
            #
            # context = attention_energies.bmm(sentence_encoding_batch)

            # decoder_input = torch.cat((context, graph_encoding), dim=2)

            sentence_batch_mask = create_mask(
                [len(s) for s in sentences],
                max_len=sentence_encoding_batch.size(1),
                device=self.device
            )

            mlp_input = self.attention(sentence_encoding_batch, mask=sentence_batch_mask)

            # mlp_input, decoder_hidden_states = self.decoder(
            #     decoder_input,
            #     prev_decoder_hidden_state_batch,
            #     prev_decoder_cell_state_batch,
            #     batch_size=batch_size
            # )
            #
            # mlp_input = mlp_input.view(batch_size, -1)

        # Use output and feed it into MLP
        transitions_output = self.transition_mlp(mlp_input)
        relations_output = self.relation_mlp(mlp_input)

        return transitions_output, relations_output#, decoder_hidden_states
