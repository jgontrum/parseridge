import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from parseridge.parser.modules.add_and_norm_layer import AddAndNormLayer
from parseridge.parser.modules.attention.positional_encodings import PositionalEncoder
from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.external_embeddings import ExternalEmbeddings
from parseridge.parser.modules.utils import get_mask


class InputEncoder(Module):
    INPUT_ENCODER_MODES = ["lstm", "transformer", "none"]

    def __init__(
        self,
        token_vocabulary,
        token_embedding_size,
        hidden_size=125,
        layers=2,
        heads=10,
        dropout=0.33,
        max_sentence_length=100,
        sum_directions=True,
        reduce_dimensionality=0,
        mode="lstm",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.token_vocabulary = token_vocabulary
        self.input_size = token_embedding_size
        self.hidden_size = hidden_size
        self.max_sentence_length = max_sentence_length
        self.sum_directions = sum_directions
        self.reduce_dimensionality = reduce_dimensionality

        self.mode = mode

        self.token_embeddings = nn.Embedding(
            num_embeddings=len(self.token_vocabulary),
            embedding_dim=token_embedding_size,
            padding_idx=self.token_vocabulary.get_id("<<<PADDING>>>"),
        )

        if self.reduce_dimensionality:
            self.dimensionality_reducer = nn.Sequential(
                nn.Linear(token_embedding_size, self.reduce_dimensionality), nn.PReLU()
            )

            self.input_size = self.reduce_dimensionality

        # TODO Add other feature embeddings here

        if self.mode == "lstm":
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=layers,
                dropout=dropout,
                bidirectional=True,
                batch_first=True,
            )

            self.output_size = hidden_size if self.sum_directions else 2 * hidden_size
        elif self.mode == "transformer":
            self.positional_encoder = PositionalEncoder(
                model_size=self.input_size, max_length=100
            )

            self.multihead_attention = MultiheadAttention(
                embed_dim=self.input_size, num_heads=heads
            )

            self.output_size = self.input_size

        elif self.mode == "none":
            self.output_size = 128

            self.embedding_transform_upscale = nn.Sequential(
                nn.Linear(token_embedding_size, self.output_size)
            )

            self.embedding_transform = nn.Sequential(
                nn.Linear(self.output_size, self.output_size), nn.ReLU()
            )

            self.embedding_transform_norm = AddAndNormLayer(model_size=self.output_size)

        else:
            raise ValueError(f"'{self.mode}' not in {self.INPUT_ENCODER_MODES}.")

    def load_external_embeddings(self, embeddings: ExternalEmbeddings):
        self.logger.info("Loading external embeddings into the embedding layer...")
        self.token_embeddings.weight = embeddings.get_weight_matrix(
            self.token_vocabulary, self.device
        )

    def forward(self, sentence_batch, sentence_lengths):
        tokens_embedded = self.token_embeddings(sentence_batch)

        if self.reduce_dimensionality:
            tokens_embedded = self.dimensionality_reducer(tokens_embedded)

        if self.mode == "lstm":
            input_packed = pack_padded_sequence(
                tokens_embedded, lengths=sentence_lengths, batch_first=True
            )

            packed_outputs, hidden = self.rnn(input_packed)

            outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

            if self.sum_directions:
                outputs = (
                    outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
                )  # Sum bidirectional outputs

            return outputs, hidden

        elif self.mode == "transformer":
            # Get an inverted mask, where '1' indicates padding
            mask = ~get_mask(
                batch=sentence_batch, lengths=sentence_lengths, device=self.device
            )

            # Add positional encodings
            tokens_embedded = self.positional_encoder(tokens_embedded)

            # [Batch, Sequence, Embedding] -> [Sequence, Batch, Embedding]
            tokens_embedded = tokens_embedded.transpose(0, 1)

            # Compute the multihead attention
            attention_output, attention_weights = self.multihead_attention(
                query=tokens_embedded,
                key=tokens_embedded,
                value=tokens_embedded,
                key_padding_mask=mask,
            )

            # [Sequence, Batch, Embedding] -> [Batch, Sequence, Embedding]
            attention_output = attention_output.transpose(0, 1)

            # Mask out padding tokens
            attention_output[mask] = float("-inf")

            return attention_output, attention_weights

        elif self.mode == "none":
            upscaled = self.embedding_transform_upscale(tokens_embedded)

            output = self.embedding_transform_norm(
                input=upscaled, output=self.embedding_transform(upscaled)
            )

            return output, None

        # TODO residual connections
        # TODO dropout
        # TODO layernorm
