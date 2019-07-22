from typing import Optional, List

import torch
from torch import Tensor

from parseridge.parser.evaluation.callbacks.attention_reporter_callback import (
    AttentionReporter,
)
from parseridge.parser.modules.attention.positional_encodings import PositionalEncoder
from parseridge.parser.modules.attention.soft_attention import Attention
from parseridge.parser.modules.attention.universal_attention import UniversalAttention
from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.utils import (
    get_padded_tensors_for_indices,
    lookup_tensors_for_indices,
)


class StaticConfigurationEncoder(Module):
    def __init__(self, model_size: int, num_stack: int = 3, num_buffer: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.input_size = model_size
        self.output_size = (num_stack + num_buffer) * model_size
        self.num_stack = num_stack
        self.num_buffer = num_buffer

    def forward(
        self,
        contextualized_input_batch: Tensor,
        stacks: Tensor,
        buffers: Tensor,
        stack_lengths: Tensor,
        buffer_lengths: Tensor,
        padding: Tensor,
        **kwargs,
    ) -> Tensor:
        stack_batch = get_padded_tensors_for_indices(
            indices=stacks,
            lengths=stack_lengths,
            contextualized_input_batch=contextualized_input_batch,
            max_length=self.num_stack,
            padding=padding,
            device=self.device,
        )

        buffer_batch = get_padded_tensors_for_indices(
            indices=buffers,
            lengths=buffer_lengths,
            contextualized_input_batch=contextualized_input_batch,
            max_length=self.num_buffer,
            padding=padding,
            device=self.device,
        )

        return torch.cat((stack_batch, buffer_batch), dim=1)


class UniversalConfigurationEncoder(Module):
    def __init__(
        self,
        model_size: int,
        scale_query: int = None,
        scale_key: int = None,
        scale_value: int = None,
        scoring_function: str = "dot",
        normalization_function: str = "softmax",
        reporter: Optional[AttentionReporter] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reporter = reporter

        self.model_size = self.input_size = model_size

        self.positional_encoder = PositionalEncoder(
            model_size=self.model_size, max_length=350
        )

        self.stack_attention = UniversalAttention(
            query_dim=self.model_size,
            similarity=scoring_function,
            normalization=normalization_function,
            query_output_dim=scale_query,
            key_output_dim=scale_key,
            value_output_dim=scale_value,
            device=self.device,
        )

        self.buffer_attention = UniversalAttention(
            query_dim=self.model_size,
            similarity=scoring_function,
            normalization=normalization_function,
            query_output_dim=scale_query,
            key_output_dim=scale_key,
            value_output_dim=scale_value,
            device=self.device,
        )

        self.output_size = (
            self.stack_attention.output_size + self.buffer_attention.output_size
        )

    def forward(
        self,
        contextualized_input_batch: Tensor,
        stacks: Tensor,
        buffers: Tensor,
        stack_lengths: Tensor,
        buffer_lengths: Tensor,
        padding: Optional[Tensor] = None,
        sentence_features: Optional[torch.Tensor] = None,
        sentence_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> Tensor:
        stack_batch = lookup_tensors_for_indices(stacks, contextualized_input_batch)
        buffer_batch = lookup_tensors_for_indices(buffers, contextualized_input_batch)

        stack_batch = self.positional_encoder(stack_batch)
        buffer_batch = self.positional_encoder(buffer_batch)

        # Compute a representation of the stack / buffer as an weighted average based
        # on the attention weights.
        stack_batch_attention, _, stack_attention_energies = self.stack_attention(
            keys=stack_batch, sequence_lengths=stack_lengths
        )

        buffer_batch_attention, _, buffer_attention_energies = self.buffer_attention(
            keys=buffer_batch, sequence_lengths=buffer_lengths
        )

        if self.reporter:
            self.reporter.log(
                "buffer",
                buffers,
                buffer_lengths,
                buffer_attention_energies,
                sentence_features,
                sentence_ids,
            )
            self.reporter.log(
                "stack",
                stacks,
                stack_lengths,
                stack_attention_energies,
                sentence_features,
                sentence_ids,
            )

        return torch.cat((stack_batch_attention, buffer_batch_attention), dim=1)


class StackBufferQueryConfigurationEncoder(Module):
    def __init__(
        self,
        model_size: int,
        scale_query: int = None,
        scale_key: int = None,
        scale_value: int = None,
        scoring_function: str = "dot",
        normalization_function: str = "softmax",
        reporter: Optional[AttentionReporter] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reporter = reporter

        self.model_size = self.input_size = model_size

        self.positional_encoder = PositionalEncoder(
            model_size=self.model_size, max_length=350
        )

        self.stack_attention = Attention(
            query_dim=self.model_size,
            key_dim=self.model_size,
            similarity=scoring_function,
            normalization=normalization_function,
            query_output_dim=scale_query,
            key_output_dim=scale_key,
            value_output_dim=scale_value,
            device=self.device,
        )

        self.buffer_attention = Attention(
            query_dim=self.model_size,
            key_dim=self.model_size,
            similarity=scoring_function,
            normalization=normalization_function,
            query_output_dim=scale_query,
            key_output_dim=scale_key,
            value_output_dim=scale_value,
            device=self.device,
        )

        self.output_size = (
            self.stack_attention.output_size + self.buffer_attention.output_size
        )

    def _fix_empty_sequence(self, tensor):
        if tensor.size(1) == 0:
            tensor = torch.zeros(
                (tensor.size(0), 1, self.model_size),
                device=self.device,
                requires_grad=False,
            )
        return tensor

    def forward(
        self,
        contextualized_input_batch: Tensor,
        stacks: Tensor,
        buffers: Tensor,
        stack_lengths: Tensor,
        buffer_lengths: Tensor,
        padding: Optional[Tensor] = None,
        sentence_features: Optional[torch.Tensor] = None,
        sentence_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> Tensor:
        # Look-up the whole unpadded buffer and stack sequence
        stack_keys = lookup_tensors_for_indices(stacks, contextualized_input_batch)
        buffer_keys = lookup_tensors_for_indices(buffers, contextualized_input_batch)

        stack_keys = self._fix_empty_sequence(stack_keys)
        buffer_keys = self._fix_empty_sequence(buffer_keys)

        # Add positional encoding
        stack_keys = self.positional_encoder(stack_keys)
        buffer_keys = self.positional_encoder(buffer_keys)

        # Take the first entry as query
        stack_queries = buffer_keys.index_select(
            dim=1, index=torch.zeros(1, dtype=torch.int64, device=self.device)
        ).squeeze(1)

        buffer_queries = stack_keys.index_select(
            dim=1, index=torch.zeros(1, dtype=torch.int64, device=self.device)
        ).squeeze(1)

        # Compute a representation of the stack / buffer as an weighted average based
        # on the attention weights.
        stack_batch_attention, _, stack_attention_energies = self.stack_attention(
            queries=stack_queries, keys=stack_keys, sequence_lengths=stack_lengths
        )

        buffer_batch_attention, _, buffer_attention_energies = self.buffer_attention(
            queries=buffer_queries, keys=buffer_keys, sequence_lengths=buffer_lengths
        )

        if self.reporter:
            self.reporter.log(
                "stack",
                stacks,
                stack_lengths,
                stack_attention_energies,
                sentence_features,
                sentence_ids,
            )
            self.reporter.log(
                "buffer",
                buffers,
                buffer_lengths,
                buffer_attention_energies,
                sentence_features,
                sentence_ids,
            )

        return torch.cat((stack_batch_attention, buffer_batch_attention), dim=1)


class FinishedQueryConfigurationEncoder(Module):
    def __init__(
        self,
        model_size: int,
        scale_query: int = None,
        scale_key: int = None,
        scale_value: int = None,
        scoring_function: str = "dot",
        normalization_function: str = "softmax",
        reporter: Optional[AttentionReporter] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reporter = reporter

        self.model_size = self.input_size = model_size

        self.positional_encoder = PositionalEncoder(
            model_size=self.model_size, max_length=350
        )

        self.finished_tokens_attention = UniversalAttention(
            query_dim=self.model_size,
            similarity=scoring_function,
            normalization=normalization_function,
            query_output_dim=scale_query,
            key_output_dim=scale_key,
            value_output_dim=scale_value,
            device=self.device,
        )

        self.stack_attention = Attention(
            query_dim=self.finished_tokens_attention.output_size,
            key_dim=self.model_size,
            similarity=scoring_function,
            normalization=normalization_function,
            query_output_dim=scale_query,
            key_output_dim=scale_key,
            value_output_dim=scale_value,
            device=self.device,
        )

        self.buffer_attention = Attention(
            query_dim=self.finished_tokens_attention.output_size,
            key_dim=self.model_size,
            similarity=scoring_function,
            normalization=normalization_function,
            query_output_dim=scale_query,
            key_output_dim=scale_key,
            value_output_dim=scale_value,
            device=self.device,
        )

        self.output_size = (
            self.stack_attention.output_size + self.buffer_attention.output_size
        )

    def _fix_empty_sequence(self, tensor):
        if tensor.size(1) == 0:
            tensor = torch.zeros(
                (tensor.size(0), 1, self.model_size),
                device=self.device,
                requires_grad=False,
            )
        return tensor

    def forward(
        self,
        contextualized_input_batch: Tensor,
        stacks: Tensor,
        buffers: Tensor,
        stack_lengths: Tensor,
        buffer_lengths: Tensor,
        finished_tokens: Tensor,
        finished_tokens_lengths: Tensor,
        padding: Optional[Tensor] = None,
        sentence_features: Optional[torch.Tensor] = None,
        sentence_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> Tensor:
        # Look-up the whole unpadded buffer and stack sequence
        stack_keys = lookup_tensors_for_indices(stacks, contextualized_input_batch)
        buffer_keys = lookup_tensors_for_indices(buffers, contextualized_input_batch)
        finished_tokens_keys = lookup_tensors_for_indices(
            finished_tokens, contextualized_input_batch
        )

        # Add a padding vector so that even empty sequences have at least one item
        stack_keys = self._fix_empty_sequence(stack_keys)
        buffer_keys = self._fix_empty_sequence(buffer_keys)
        finished_tokens_keys = self._fix_empty_sequence(finished_tokens_keys)

        # Add positional encoding
        stack_keys = self.positional_encoder(stack_keys)
        buffer_keys = self.positional_encoder(buffer_keys)
        finished_tokens_keys = self.positional_encoder(finished_tokens_keys)

        # Run universal attention over the finished tokens
        tokens_attention, _, tokens_attention_energies = self.finished_tokens_attention(
            keys=finished_tokens_keys, sequence_lengths=finished_tokens_lengths
        )

        # Compute a representation of the stack / buffer as an weighted average based
        # on the attention weights.
        stack_batch_attention, _, stack_attention_energies = self.stack_attention(
            queries=tokens_attention, keys=stack_keys, sequence_lengths=stack_lengths
        )

        buffer_batch_attention, _, buffer_attention_energies = self.buffer_attention(
            queries=tokens_attention, keys=buffer_keys, sequence_lengths=buffer_lengths
        )

        if self.reporter:
            self.reporter.log(
                "finished_tokens",
                finished_tokens,
                finished_tokens_lengths,
                tokens_attention_energies,
                sentence_features,
                sentence_ids,
            )
            self.reporter.log(
                "stack",
                stacks,
                stack_lengths,
                stack_attention_energies,
                sentence_features,
                sentence_ids,
            )
            self.reporter.log(
                "buffer",
                buffers,
                buffer_lengths,
                buffer_attention_energies,
                sentence_features,
                sentence_ids,
            )

        return torch.cat((stack_batch_attention, buffer_batch_attention), dim=1)


class SentenceQueryConfigurationEncoder(Module):
    def __init__(
        self,
        model_size: int,
        scale_query: int = None,
        scale_key: int = None,
        scale_value: int = None,
        scoring_function: str = "dot",
        normalization_function: str = "softmax",
        reporter: Optional[AttentionReporter] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reporter = reporter

        self.model_size = self.input_size = model_size

        self.positional_encoder = PositionalEncoder(
            model_size=self.model_size, max_length=350
        )

        self.finished_tokens_attention = UniversalAttention(
            query_dim=self.model_size,
            similarity=scoring_function,
            normalization=normalization_function,
            query_output_dim=scale_query,
            key_output_dim=scale_key,
            value_output_dim=scale_value,
            device=self.device,
        )

        self.sentence_attention = Attention(
            query_dim=self.finished_tokens_attention.output_size,
            key_dim=self.model_size,
            similarity=scoring_function,
            normalization=normalization_function,
            query_output_dim=scale_query,
            key_output_dim=scale_key,
            value_output_dim=scale_value,
            device=self.device,
        )

        self.stack_attention = Attention(
            query_dim=self.sentence_attention.output_size,
            key_dim=self.model_size,
            similarity=scoring_function,
            normalization=normalization_function,
            query_output_dim=scale_query,
            key_output_dim=scale_key,
            value_output_dim=scale_value,
            device=self.device,
        )

        self.buffer_attention = Attention(
            query_dim=self.sentence_attention.output_size,
            key_dim=self.model_size,
            similarity=scoring_function,
            normalization=normalization_function,
            query_output_dim=scale_query,
            key_output_dim=scale_key,
            value_output_dim=scale_value,
            device=self.device,
        )

        self.output_size = (
            self.stack_attention.output_size + self.buffer_attention.output_size
        )

    def _fix_empty_sequence(self, tensor):
        if tensor.size(1) == 0:
            tensor = torch.zeros(
                (tensor.size(0), 1, self.model_size),
                device=self.device,
                requires_grad=False,
            )
        return tensor

    def forward(
        self,
        contextualized_input_batch: Tensor,
        stacks: Tensor,
        buffers: Tensor,
        stack_lengths: Tensor,
        buffer_lengths: Tensor,
        finished_tokens: Tensor,
        finished_tokens_lengths: Tensor,
        sentence_lengths: Tensor,
        padding: Optional[Tensor] = None,
        sentence_features: Optional[torch.Tensor] = None,
        sentence_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> Tensor:
        # Look-up the whole unpadded buffer and stack sequence
        stack_keys = lookup_tensors_for_indices(stacks, contextualized_input_batch)
        buffer_keys = lookup_tensors_for_indices(buffers, contextualized_input_batch)
        finished_tokens_keys = lookup_tensors_for_indices(
            finished_tokens, contextualized_input_batch
        )

        # Add a padding vector so that even empty sequences have at least one item
        stack_keys = self._fix_empty_sequence(stack_keys)
        buffer_keys = self._fix_empty_sequence(buffer_keys)
        finished_tokens_keys = self._fix_empty_sequence(finished_tokens_keys)

        # Add positional encoding
        stack_keys = self.positional_encoder(stack_keys)
        buffer_keys = self.positional_encoder(buffer_keys)
        finished_tokens_keys = self.positional_encoder(finished_tokens_keys)
        sentence_keys = self.positional_encoder(contextualized_input_batch)

        # Run universal attention over the finished tokens
        tokens_attention, _, tokens_attention_energies = self.finished_tokens_attention(
            keys=finished_tokens_keys, sequence_lengths=finished_tokens_lengths
        )

        sentence_attention, _, sentence_attention_energies = self.sentence_attention(
            queries=tokens_attention, keys=sentence_keys, sequence_lengths=sentence_lengths
        )

        # Compute a representation of the stack / buffer as an weighted average based
        # on the attention weights.
        stack_batch_attention, _, stack_attention_energies = self.stack_attention(
            queries=sentence_attention, keys=stack_keys, sequence_lengths=stack_lengths
        )

        buffer_batch_attention, _, buffer_attention_energies = self.buffer_attention(
            queries=sentence_attention, keys=buffer_keys, sequence_lengths=buffer_lengths
        )

        if self.reporter:
            sentence_tokens = torch.arange(
                contextualized_input_batch.size(1), device=self.device
            ).expand(contextualized_input_batch.size(0), contextualized_input_batch.size(1))
            self.reporter.log(
                "finished_tokens",
                finished_tokens,
                finished_tokens_lengths,
                tokens_attention_energies,
                sentence_features,
                sentence_ids,
            )
            self.reporter.log(
                "sentence",
                sentence_tokens,
                sentence_lengths,
                sentence_attention_energies,
                sentence_features,
                sentence_ids,
            )
            self.reporter.log(
                "stack",
                stacks,
                stack_lengths,
                stack_attention_energies,
                sentence_features,
                sentence_ids,
            )
            self.reporter.log(
                "buffer",
                buffers,
                buffer_lengths,
                buffer_attention_energies,
                sentence_features,
                sentence_ids,
            )

        return torch.cat((stack_batch_attention, buffer_batch_attention), dim=1)


CONFIGURATION_ENCODERS = {
    "static": StaticConfigurationEncoder,
    "universal_attention": UniversalConfigurationEncoder,
    "stack-buffer_query_attention": StackBufferQueryConfigurationEncoder,
    "finished_tokens_attention": FinishedQueryConfigurationEncoder,
    "sentence_query_attention": SentenceQueryConfigurationEncoder,
}
