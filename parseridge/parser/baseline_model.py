from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from parseridge.corpus.relations import Relations
from parseridge.corpus.vocabulary import Vocabulary
from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.input_encoder import InputEncoder
from parseridge.parser.modules.mlp import MultilayerPerceptron
from parseridge.parser.modules.utils import initialize_xavier_dynet_, \
    lookup_tensors_for_indices, mask_, pad_tensor_list


class BaselineModel(Module):

    def __init__(self,
                 relations: Relations,
                 vocabulary: Vocabulary,
                 num_stack: int = 3,
                 num_buffer: int = 1,
                 lstm_dropout: float = 0.33,
                 mlp_dropout: float = 0.25,
                 embedding_size: int = 100,
                 lstm_hidden_size: int = 125,
                 lstm_layers: int = 2,
                 transition_mlp_layers: List[int] = None,
                 relation_mlp_layers: List[int] = None,
                 embeddings: List[int] = None,
                 device: str = "cpu") -> None:

        super().__init__(device=device)

        """ Parameter definitions """
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

        self.input_encoder = InputEncoder(
            token_vocabulary=vocabulary,
            token_embedding_size=embedding_size,
            hidden_size=lstm_hidden_size,
            layers=lstm_layers,
            dropout=lstm_dropout,
            positional_embedding_size=False,
            sum_directions=False,
            reduce_dimensionality=False,
            device=device
        )

        self.mlp_in_size = (
                (self.stack_size + self.buffer_size) *
                self.input_encoder.output_size
        )

        self.transition_mlp = MultilayerPerceptron(
            input_size=self.mlp_in_size,
            hidden_sizes=transition_mlp_layers,
            output_size=self.num_transitions,
            dropout=self.mlp_dropout,
            activation=nn.Tanh,
            device=device
        )

        self.relation_mlp = MultilayerPerceptron(
            input_size=self.mlp_in_size,
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

        if embeddings:
            self.input_encoder.load_external_embeddings(embeddings)

    def reset_(self) -> None:
        self._mlp_padding = nn.Tanh()(self._mlp_padding_param)

    def get_contextualized_input(self,
                                 token_sequences: Tensor,
                                 sentence_lengths: Tensor) -> Tensor:
        outputs, _ = self.input_encoder(token_sequences, sentence_lengths)
        return outputs

    def compute_mlp_output(self,
                           contextualized_input_batch: Tensor,
                           stacks: Tensor,
                           buffers: Tensor,
                           stack_lengths: Tensor,
                           buffer_lengths: Tensor,
                           ) -> Tuple[Tensor, Tensor]:

        # Lookup the contextualized tokens from the indices
        stack_batch = self._get_padded_tensors_for_indices(
            indices=stacks,
            lengths=stack_lengths,
            contextualized_input_batch=contextualized_input_batch,
            max_length=self.stack_size
        )

        buffer_batch = self._get_padded_tensors_for_indices(
            indices=buffers,
            lengths=buffer_lengths,
            contextualized_input_batch=contextualized_input_batch,
            max_length=self.buffer_size
        )

        mlp_input = torch.cat((stack_batch, buffer_batch), dim=1)

        transitions_output = self.transition_mlp(mlp_input)
        relations_output = self.relation_mlp(mlp_input)

        return transitions_output, relations_output

    def _get_padded_tensors_for_indices(self, indices: Tensor, lengths: Tensor,
                                        contextualized_input_batch: Tensor, max_length:int):
        indices = pad_tensor_list(indices, length=max_length)
        # Lookup the contextualized tokens from the indices
        batch = lookup_tensors_for_indices(indices, contextualized_input_batch)

        batch_size = batch.size(0)
        sequence_size = max(batch.size(1), max_length)
        token_size = batch.size(2)

        # Expand the padding vector over the size of the batch
        padding_batch = self._mlp_padding.expand(batch_size, sequence_size, token_size)

        if max(lengths) == 0:
            # If the batch is completely empty, we can just return the whole padding batch
            batch_padded = padding_batch
        else:
            # Build a mask and expand it over the size of the batch
            mask = torch.arange(sequence_size)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(2).expand(batch_size, sequence_size, token_size)

            batch_padded = torch.where(
                mask,  # Condition
                batch,  # If condition is 1
                padding_batch  # If condition is 0
            )

            # Cut the tensor at the specified length
            batch_padded = torch.split(batch_padded, max_length, dim=1)[0]

        # Flatten the output by concatenating the token embeddings
        return batch_padded.contiguous().view(batch_padded.size(0), -1)

    def forward(self,
                stacks: Tensor,
                buffers: Tensor,
                stack_lengths: Optional[Tensor] = None,
                buffer_lengths: Optional[Tensor] = None,
                token_sequences: Optional[Tensor] = None,
                sentence_lengths: Optional[Tensor] = None,
                contextualized_input_batch: Optional[List[Tensor]] = None
                ) -> Tuple[Tensor, Tensor]:

        if contextualized_input_batch is None:
            assert token_sequences is not None
            assert sentence_lengths is not None
            # Pass all sentences through the input encoder to create contextualized
            # token tensors.
            contextualized_input_batch = self.compute_lstm_output(
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
            buffer_lengths=buffer_lengths
        )

        return transitions_output, relations_output
