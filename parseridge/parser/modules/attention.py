import torch
import torch.nn as nn

from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.utils import initialize_xavier_dynet_


class Attention(Module):
    def __init__(self, input_size: int, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.input_size = input_size
        self.output_size = input_size

        self.word_weight = nn.Sequential(
            nn.Linear(
                in_features=input_size,
                out_features=input_size
            ),
            nn.Tanh()
        )

        # TODO add cosine or dot product here instead of learning the function
        self.context_weight = nn.Linear(
            in_features=input_size,
            out_features=1
        )

        initialize_xavier_dynet_(self)

    def forward(self, input: torch.Tensor, mask: torch.Tensor = None, debug=False) -> torch.Tensor:
        word_representations = self.word_weight(input)

        similarity_scores_logits = self.context_weight(word_representations)

        if mask is not None:
            mask = mask.unsqueeze(2)
            similarity_scores_logits.masked_fill_(mask, -1e18)

        activation_energies = torch.softmax(similarity_scores_logits, dim=1)

        weighted_input = input * activation_energies

        sentence_vector = torch.sum(weighted_input, dim=1)

        if debug:
            return sentence_vector, activation_energies

        return sentence_vector
