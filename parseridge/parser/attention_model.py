from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from parseridge.corpus.relations import Relations
from parseridge.corpus.vocabulary import Vocabulary
from parseridge.parser.modules.attention.universal_attention import UniversalAttention
from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.external_embeddings import ExternalEmbeddings
from parseridge.parser.modules.input_encoder import InputEncoder
from parseridge.parser.modules.mlp import MultilayerPerceptron
from parseridge.parser.modules.utils import (
    initialize_xavier_dynet_,
    lookup_tensors_for_indices,
)


class AttentionModel(Module):
    def __init__(
        self,
        relations: Relations,
        vocabulary: Vocabulary,
        num_stack: int = 3,
        num_buffer: int = 1,
        lstm_dropout: float = 0.33,
        mlp_dropout: float = 0.25,
        embedding_size: int = 100,
        lstm_hidden_size: int = 125,
        lstm_layers: int = 2,
        input_encoder_type: str = "lstm",
        transition_mlp_layers: List[int] = None,
        relation_mlp_layers: List[int] = None,
        transition_mlp_activation: nn.Module = nn.Tanh,
        relation_mlp_activation: nn.Module = nn.Tanh,
        embeddings: ExternalEmbeddings = None,
        self_attention_heads: int = 10,
        scale_query: int = None,
        scale_key: int = None,
        scale_value: int = None,
        scoring_function: str = "dot",
        normalization_function: str = "softmax",
        device: str = "cpu",
    ) -> None:

        super().__init__(device=device)

        """ Parameter definitions """
        self.relations = relations
        self.vocabulary = vocabulary

        self.input_encoder_type = input_encoder_type

        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_dropout = lstm_dropout
        self.lstm_layers = lstm_layers

        self.mlp_dropout = mlp_dropout

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

        self.input_encoder = InputEncoder(
            token_vocabulary=self.vocabulary,
            token_embedding_size=self.embedding_size,
            hidden_size=self.lstm_hidden_size,
            layers=self.lstm_layers,
            dropout=self.lstm_dropout,
            sum_directions=False,
            reduce_dimensionality=False,
            mode=self.input_encoder_type,
            heads=self_attention_heads,
            device=self.device,
        )

        """Computes attention over the output of the input encoder given the state of the
        action encoder. """
        self.stack_attention = UniversalAttention(
            query_dim=self.input_encoder.output_size,
            similarity=scoring_function,
            normalization=normalization_function,
            query_output_dim=scale_query,
            key_output_dim=scale_key,
            value_output_dim=scale_value,
            device=device,
        )

        self.buffer_attention = UniversalAttention(
            query_dim=self.input_encoder.output_size,
            similarity=scoring_function,
            normalization=normalization_function,
            query_output_dim=scale_query,
            key_output_dim=scale_key,
            value_output_dim=scale_value,
            device=device,
        )

        self.mlp_in_size = (
            self.buffer_attention.output_size + self.stack_attention.output_size
        )

        self.transition_mlp = MultilayerPerceptron(
            input_size=self.mlp_in_size,
            hidden_sizes=transition_mlp_layers,
            output_size=self.num_transitions,
            dropout=self.mlp_dropout,
            activation=transition_mlp_activation,
            device=self.device,
        )

        self.relation_mlp = MultilayerPerceptron(
            input_size=self.mlp_in_size,
            hidden_sizes=relation_mlp_layers,
            output_size=self.num_labels,
            dropout=self.mlp_dropout,
            activation=relation_mlp_activation,
            device=self.device,
        )

        self._mlp_padding_param = nn.Parameter(
            torch.zeros(self.input_encoder.output_size, dtype=torch.float)
        )
        self._mlp_padding = None

        initialize_xavier_dynet_(self)

        if embeddings:
            self.input_encoder.load_external_embeddings(embeddings)

    def reset_(self) -> None:
        self._mlp_padding = nn.Tanh()(self._mlp_padding_param)

    def get_contextualized_input(
        self, token_sequences: torch.Tensor, sentence_lengths: torch.Tensor
    ) -> torch.Tensor:
        outputs, _ = self.input_encoder(token_sequences, sentence_lengths)
        return outputs

    def compute_mlp_output(
        self,
        contextualized_input_batch: torch.Tensor,
        stacks: torch.Tensor,
        stack_lengths: torch.Tensor,
        buffers: torch.Tensor,
        buffer_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stacks = lookup_tensors_for_indices(stacks, contextualized_input_batch)
        buffers = lookup_tensors_for_indices(buffers, contextualized_input_batch)

        # Compute a representation of the stack / buffer as an weighted average based
        # on the attention weights.
        stack_batch_attention, _, stack_attention_energies = self.stack_attention(
            keys=stacks, sequence_lengths=stack_lengths
        )

        buffer_batch_attention, _, buffer_attention_energies = self.buffer_attention(
            keys=buffers, sequence_lengths=buffer_lengths
        )

        mlp_input = torch.cat((stack_batch_attention, buffer_batch_attention), dim=1)

        # Use output and feed it into MLP
        transitions_output = self.transition_mlp(mlp_input)
        relations_output = self.relation_mlp(mlp_input)

        return transitions_output, relations_output

    def forward(
        self,
        stacks: torch.Tensor,
        stack_lengths: torch.Tensor,
        buffers: torch.Tensor,
        buffer_lengths: torch.Tensor,
        token_sequences: Optional[torch.Tensor] = None,
        sentence_lengths: Optional[torch.Tensor] = None,
        contextualized_input_batch: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if contextualized_input_batch is None:
            assert token_sequences is not None
            assert sentence_lengths is not None
            # Pass all sentences through the input encoder to create contextualized
            # token tensors.
            contextualized_input_batch = self.get_contextualized_input(
                token_sequences, sentence_lengths
            )
        else:
            # If we already have the output of the input encoder as lists,
            # create a tensor out of them. This happens usually in prediction.
            contextualized_input_batch = torch.stack(contextualized_input_batch)

        transitions_output, relations_output = self.compute_mlp_output(
            contextualized_input_batch=contextualized_input_batch,
            stacks=stacks,
            stack_lengths=stack_lengths,
            buffers=buffers,
            buffer_lengths=buffer_lengths,
        )

        return transitions_output, relations_output
