import os
from typing import Any, Optional, List

from parseridge.parser.evaluation.callbacks.base_eval_callback import EvalCallback


class EvalSaveParsedSentencesCallback(EvalCallback):
    _order = 1000

    def __init__(self, output_dir_path: Optional[str] = None) -> None:
        self.output_dir_path = output_dir_path
        self.current_epoch = 0

    def on_epoch_end(
        self, pred_sentences_serialized: List[str], corpus_type: str, **kwargs: Any
    ) -> None:
        if self.output_dir_path:
            filename = self.output_dir_path.rstrip("/")
            filename += f"/epoch_{self.current_epoch}-{corpus_type}.conllu"

            os.makedirs(os.path.dirname(filename), exist_ok=True)

            with open(filename, "w") as f:
                f.write("".join(pred_sentences_serialized))

    def on_eval_begin(self, epoch: int, **kwargs) -> None:
        self.current_epoch = epoch
