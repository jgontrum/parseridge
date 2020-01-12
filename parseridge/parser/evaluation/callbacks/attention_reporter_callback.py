import json
import lzma
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Optional

from torch import Tensor

from parseridge.corpus.vocabulary import Vocabulary
from parseridge.parser.evaluation.callbacks.base_eval_callback import EvalCallback


@dataclass
class EvalAttentionReporter(EvalCallback):
    file_path: Optional[str]
    vocabulary: Vocabulary

    def __post_init__(self):
        self._current_epoch = 1
        self._reset()
        if self.file_path:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def _reset(self):
        if self.file_path:
            self._current_data = defaultdict(lambda: defaultdict(list))

    def _save(self, file_name):
        if self.file_path:
            self.logger.info(f"Saving compressed attention weights to '{file_name}'.")
            with lzma.open(file_name, mode="w") as f:
                data = json.dumps(self._current_data)
                f.write(data.encode())

    def log(
        self,
        name: str,
        sequences: Tensor,
        sequence_lengths: Tensor,
        attention_energies: Tensor,
        sentence_features: Tensor,
        sentence_ids: List[str],
    ):
        sentences_token_ids = sentence_features[:, 0, :]

        for sequence, length, energies, sentence_token_ids, sentence_id in zip(
            sequences,
            sequence_lengths,
            attention_energies,
            sentences_token_ids,
            sentence_ids,
        ):
            tokens = sentence_token_ids.index_select(dim=0, index=sequence).cpu().tolist()
            tokens = [self.vocabulary.get_item(token_id) for token_id in tokens]
            energies = energies.squeeze(1).cpu().tolist()

            length = length.item()

            aligned = [
                (token, energy)
                for i, (token, energy) in enumerate(zip(tokens, energies))
                if i < length
            ]

            self._current_data[sentence_id][name].append(aligned)

    def on_eval_begin(self, **kwargs: Any) -> None:
        self._save(
            f"{self.file_path}/attention_weights_train_epoch_{self._current_epoch}.json.lzma"
        )
        self._reset()

    def on_eval_end(self, **kwargs: Any) -> None:
        self._reset()
        self._current_epoch += 1
