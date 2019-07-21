import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List

from torch import Tensor

from parseridge.corpus.vocabulary import Vocabulary
from parseridge.parser.evaluation.callbacks.base_eval_callback import EvalCallback


@dataclass
class AttentionReporter(EvalCallback):
    file_path: str
    vocabulary: Vocabulary

    def __post_init__(self):
        self._current_epoch = 1
        self._reset()
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def _reset(self):
        self._current_data = defaultdict(lambda: defaultdict(list))

    def log(
        self,
        name: str,
        sequences: Tensor,
        attention_energies: Tensor,
        sentence_features: Tensor,
        sentence_ids: List[str],
    ):
        sentences_token_ids = sentence_features[:, 0, :]

        for sequence, energies, sentence_token_ids, sentence_id in zip(
            sequences, attention_energies, sentences_token_ids, sentence_ids
        ):
            tokens = sentence_token_ids.index_select(dim=0, index=sequence).cpu().tolist()
            tokens = [self.vocabulary.get_item(token_id) for token_id in tokens]
            energies = energies.squeeze(1).cpu().tolist()

            aligned = [
                (token, energy) for token, energy in zip(tokens, energies) if energy > 0.0
            ]

            self._current_data[sentence_id][name].append(aligned)

    def on_eval_begin(self, **kwargs: Any) -> None:
        file_name = (
            f"{self.file_path}/attention_weights_train_epoch_{self._current_epoch}.json"
        )

        with open(file_name, mode="w") as f:
            json.dump(self._current_data, f)

        self._reset()

    def on_eval_end(self, **kwargs: Any) -> None:
        file_name = (
            f"{self.file_path}/attention_weights_eval_epoch_{self._current_epoch}.json"
        )

        with open(file_name, mode="w") as f:
            json.dump(self._current_data, f)

        self._reset()

        self._current_epoch += 1
