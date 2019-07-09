from typing import Any, Dict

from parseridge.parser.evaluation.callbacks.base_eval_callback import EvalCallback


class EvalSimpleLogger(EvalCallback):
    def on_eval_end(self, scores: Dict[str, Dict[str, float]], **kwargs: Any) -> None:
        self.logger.info(
            f"Evaluation on 'train': {scores['train']['las']:2f} LAS, "
            f"{scores['train']['uas']:2f} UAS."
        )

        self.logger.info(
            f"Evaluation on 'dev'  : {scores['dev']['las']:2f} LAS, "
            f"{scores['dev']['uas']:2f} UAS."
        )

        if scores["test"]["las"] is not None:
            self.logger.info(
                f"Evaluation on 'test' : {scores['test']['las']:2f} LAS, "
                f"{scores['test']['uas']:2f} UAS."
            )
