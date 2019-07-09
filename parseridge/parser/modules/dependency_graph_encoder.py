import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.utils import add_padding


class DependencyGraphEncoder(Module):
    def __init__(
        self,
        input_size,
        output_size,
        relations,
        relation_embedding_size=64,
        num_layers=1,
        device="cpu",
    ):
        super(DependencyGraphEncoder, self).__init__()
        self.device = device

        self.lstm_input_size = input_size * 3 + relation_embedding_size
        self.lstm_output_size = output_size

        self.input_size = input_size
        self.output_size = self.lstm_input_size * 2

        self.num_layers = num_layers
        self.relations = relations
        self.relation_embedding_size = relation_embedding_size

        self.relation_embedding = nn.Embedding(
            num_embeddings=len(self.relations.relations),  # Number of relation labels
            embedding_dim=self.relation_embedding_size,
        )

        self.rnn = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_output_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.token_padding = torch.zeros(self.input_size, device=self.device)
        self.graph_padding = torch.zeros(self.lstm_input_size, device=self.device)

    def get_initial_state(self, batch_size):
        # Initial state, no transition has been made so far. Return a zero filled
        # tensor and initialize the hidden state / cell state tensors.
        return (
            torch.zeros((batch_size, 1, self.output_size), device=self.device),
            self.init_hidden(batch_size),
        )

    def _get_nodes_from_sentence(self, sentence, contextualized_tokens):
        for token in sentence:
            if token.head is None and not token.dependents:
                # Skip tokens that haven't been touched yet
                continue

            token_representation = contextualized_tokens[token.id]

            head_representation = (
                contextualized_tokens[token.head]
                if token.head is not None
                else self.token_padding
            )

            assert head_representation.size()[0] == self.token_padding.size()[0]

            if token.dependents:
                dependents_representation = tuple(
                    contextualized_tokens[dependent] for dependent in token.dependents
                )

                dependents_representation = torch.sum(
                    torch.stack(dependents_representation), dim=0
                )
            else:
                dependents_representation = self.token_padding

            assert dependents_representation.size()[0] == self.token_padding.size()[0]

            relation_representation = self.relation_embedding(
                torch.tensor(
                    self.relations.signature.get_id(token.relation), device=self.device
                )
            )
            assert relation_representation.size()[0] == self.relation_embedding_size

            node_representation = (
                token_representation,
                head_representation,
                dependents_representation,
                relation_representation,
            )

            yield torch.cat(node_representation, dim=0)

    def forward(self, predicted_sentence_batch, contextualized_tokens_batch):
        partial_graph_batch = []
        for sentence, contextualized_tokens in zip(
            predicted_sentence_batch, contextualized_tokens_batch
        ):
            partial_graph = self._get_nodes_from_sentence(sentence, contextualized_tokens)
            partial_graph = list(partial_graph)
            partial_graph_batch.append(partial_graph)

        # The batch must be ordered in decreasing order. Save the original ordering here.
        order = torch.tensor(
            sorted(
                range(len(partial_graph_batch)),
                key=lambda k: len(partial_graph_batch[k]),
                reverse=True,
            ),
            device=self.device,
        )

        partial_graph_batch = sorted(
            partial_graph_batch, key=lambda graph: len(graph), reverse=True
        )

        # Pad the batch of partial graphs
        graph_lengths = [max(1, len(graph)) for graph in partial_graph_batch]
        max_graph_length = max(graph_lengths)

        if max_graph_length > 0:
            partial_graph_batch_padded = []
            for partial_graph in partial_graph_batch:
                padded = add_padding(partial_graph, max_graph_length, self.graph_padding)
                partial_graph_batch_padded.append(torch.stack(padded))

            partial_graph_batch_padded = torch.stack(partial_graph_batch_padded)

        else:
            # First iteration: All graphs are empty at this point.
            partial_graph_batch_padded = [self.graph_padding for _ in partial_graph_batch]
            partial_graph_batch_padded = torch.stack(partial_graph_batch_padded)
            partial_graph_batch_padded = partial_graph_batch_padded.unsqueeze(1)

        # Run the batch through the LSTM
        input_packed = pack_padded_sequence(
            partial_graph_batch_padded,
            torch.tensor(graph_lengths, dtype=torch.int64, device=self.device),
            batch_first=True,
        )

        packed_outputs, hidden = self.rnn(input_packed)

        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        # Get the first state of each graph. Since the RNN is bi-directional,
        # this is as good as the last state, but way easier to extract.
        last_output = outputs[:, 0, :]

        # Restore the original ordering
        return torch.index_select(last_output, dim=0, index=order)
