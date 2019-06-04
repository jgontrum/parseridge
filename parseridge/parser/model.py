import torch
import torch.nn as nn

from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.input_encoder import InputEncoder
from parseridge.parser.modules.mlp import MultilayerPerceptron
from parseridge.parser.modules.sequence_attention import SequenceAttention
from parseridge.parser.modules.utils import initialize_xavier_dynet_
from parseridge.utils.helpers import get_parameters


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
                 embeddings=None,
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
            positional_embedding_size=False,
            sum_directions=False,
            reduce_dimensionality=True,
            device=device
        )

        if embeddings:
            self.input_encoder.load_external_embeddings(embeddings)

        """Computes attention over the output of the input encoder given the state of the
        action encoder. """
        self.stack_attention = SequenceAttention(
            input_size=self.input_encoder.output_size,
            lstm_size=False,
            positional_embedding_size=64,
            device=device
        )

        self.buffer_attention = SequenceAttention(
            input_size=self.input_encoder.output_size,
            lstm_size=False,
            positional_embedding_size=64,
            device=device
        )

        self.transition_mlp = MultilayerPerceptron(
            input_size=self.stack_attention.output_size + self.buffer_attention.output_size,
            hidden_sizes=transition_mlp_layers,
            output_size=self.num_transitions,
            dropout=self.mlp_dropout,
            activation=nn.ReLU,
            device=device
        )

        self.relation_mlp = MultilayerPerceptron(
            input_size=self.transition_mlp.input_size,
            hidden_sizes=relation_mlp_layers,
            output_size=self.num_labels,
            dropout=self.mlp_dropout,
            activation=nn.ReLU,
            device=device
        )

        self._mlp_padding_param = nn.Parameter(
            torch.zeros(self.input_encoder.output_size, dtype=torch.float)
        )

        initialize_xavier_dynet_(self)
        self.logger.info(f"Learning {len(get_parameters(self))} parameters.")

    # Hooks
    def before_batch(self):
        self._mlp_padding = torch.tanh(self._mlp_padding_param)

    def after_batch(self):
        pass

    def before_epoch(self):
        self.train()
        self.zero_grad()

    def after_epoch(self):
        pass

    def compute_lstm_output(self, sentences, sentence_lengths):
        outputs, _ = self.input_encoder(sentences, sentence_lengths)
        return outputs

    def forward(self, sentences, sentence_lengths, stacks, stack_lengths,
                buffers, buffer_lengths, sentence_encoding_batch=None):

        if sentence_encoding_batch is None:
            # Pass all sentences through the input encoder to create contextualized
            # token tensors.
            sentence_encoding_batch = self.compute_lstm_output(
                sentences, sentence_lengths
            )
        else:
            # If we already have the output of the input encoder as lists,
            # create a tensor out of them. This happens usually in prediction.
            sentence_encoding_batch = torch.stack(sentence_encoding_batch)

        # Compute a representation of the stack / buffer as an weighted average based
        # on the attention weights.
        stack_batch_attn = self.stack_attention(
            indices_batch=stacks,
            indices_lengths=stack_lengths,
            sentence_encoding_batch=sentence_encoding_batch
        )

        buffer_batch_attn = self.buffer_attention(
            indices_batch=buffers,
            indices_lengths=buffer_lengths,
            sentence_encoding_batch=sentence_encoding_batch
        )

        mlp_input = torch.cat((stack_batch_attn, buffer_batch_attn), dim=1)

        # Use output and feed it into MLP
        transitions_output = self.transition_mlp(mlp_input)
        relations_output = self.relation_mlp(mlp_input)

        return transitions_output, relations_output
