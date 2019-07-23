from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from parseridge.corpus.relations import Relations
from parseridge.corpus.vocabulary import Vocabulary
from parseridge.parser.modules.configuration_encoder import (
    CONFIGURATION_ENCODERS,
    AttentionReporter,
)
from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.external_embeddings import ExternalEmbeddings
from parseridge.parser.modules.input_encoder import InputEncoder
from parseridge.parser.modules.mlp import MultilayerPerceptron
from parseridge.parser.modules.utils import initialize_xavier_dynet_


class ParseridgeModel(Module):
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
        configuration_encoder: str = "static",
        scale_query: int = None,
        scale_key: int = None,
        scale_value: int = None,
        scoring_function: str = "dot",
        normalization_function: str = "softmax",
        attention_reporter: Optional[AttentionReporter] = None,
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

        self.lstm_output_transform = MultilayerPerceptron(
            input_size=self.input_encoder.output_size,
            hidden_sizes=[512],
            output_size=self.input_encoder.output_size,
            activation=nn.ReLU,
        )

        """Computes attention over the output of the input encoder given the state of the
        action encoder. """
        self.configuration_encoder = CONFIGURATION_ENCODERS[configuration_encoder](
            model_size=self.input_encoder.output_size,
            scale_query=scale_query,
            scale_key=scale_key,
            scale_value=scale_value,
            scoring_function=scoring_function,
            normalization_function=normalization_function,
            num_stack=self.stack_size,
            num_buffer=self.buffer_size,
            reporter=attention_reporter,
            device=self.device,
        )

        self.mlp_in_size = self.configuration_encoder.output_size

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
        finished_tokens: Optional[torch.Tensor] = None,
        finished_tokens_lengths: Optional[torch.Tensor] = None,
        sentence_lengths: Optional[torch.Tensor] = None,
        sentence_features: Optional[torch.Tensor] = None,
        sentence_ids: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        contextualized_input_batch = self.lstm_output_transform(contextualized_input_batch)

        mlp_input = self.configuration_encoder(
            contextualized_input_batch=contextualized_input_batch,
            stacks=stacks,
            buffers=buffers,
            stack_lengths=stack_lengths,
            buffer_lengths=buffer_lengths,
            finished_tokens=finished_tokens,
            finished_tokens_lengths=finished_tokens_lengths,
            sentence_lengths=sentence_lengths,
            sentence_features=sentence_features,
            sentence_ids=sentence_ids,
            padding=self._mlp_padding,
        )

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
        sentence_tokens: Optional[torch.Tensor] = None,
        sentence_lengths: Optional[torch.Tensor] = None,
        contextualized_input_batch: Optional[List[torch.Tensor]] = None,
        finished_tokens: Optional[torch.Tensor] = None,
        finished_tokens_lengths: Optional[torch.Tensor] = None,
        sentence_features: Optional[torch.Tensor] = None,
        sentence_ids: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if contextualized_input_batch is None:
            assert sentence_tokens is not None
            assert sentence_lengths is not None
            # Pass all sentences through the input encoder to create contextualized
            # token tensors.
            contextualized_input_batch = self.get_contextualized_input(
                sentence_tokens, sentence_lengths
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
            sentence_lengths=sentence_lengths,
            finished_tokens=finished_tokens,
            finished_tokens_lengths=finished_tokens_lengths,
            sentence_features=sentence_features,
            sentence_ids=sentence_ids,
        )

        return transitions_output, relations_output
