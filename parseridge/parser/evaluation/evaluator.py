from dataclasses import dataclass
from types import FunctionType
from typing import List, Dict, Tuple

import torch

from parseridge.corpus.corpus import CorpusIterator, Corpus
from parseridge.corpus.treebank import Treebank
from parseridge.parser.configuration import Configuration
from parseridge.parser.evaluation.callbacks.base_eval_callback import EvalCallback
from parseridge.parser.evaluation.callbacks.handler import EvalCallbackHandler
from parseridge.parser.evaluation.conll_eval import CoNLLEvaluationScript
from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.training.dynamic_trainer import DynamicTrainer
from parseridge.utils.helpers import T
from parseridge.utils.logger import LoggerMixin


@dataclass
class Evaluator(LoggerMixin):
    model: Module
    treebank: Treebank
    callbacks: List[EvalCallback] = None
    batch_size: int = 512
    eval_function: FunctionType = CoNLLEvaluationScript().get_las_score_for_sentences

    def __post_init__(self) -> None:
        self.callback_handler = EvalCallbackHandler(callbacks=self.callbacks or [])
        self.callback_handler.on_initialization()

    def shutdown(self):
        self.callback_handler.on_shutdown()

    def evaluate(self, epoch: int = -1) -> Dict[str, Dict[str, float]]:
        self.model.eval()
        self.callback_handler.on_eval_begin(epoch=epoch)

        train_las, train_uas = self._evaluate_corpus(
            self.treebank.train_corpus, corpus_type="train")

        dev_las, dev_uas = self._evaluate_corpus(
            self.treebank.dev_corpus, corpus_type="dev")

        test_las = test_uas = None
        if self.treebank.test_corpus:
            test_las, test_uas = self._evaluate_corpus(
                self.treebank.test_corpus, corpus_type="test")

        scores = {
            "train": {
                "las": train_las,
                "uas": train_uas
            },
            "dev": {
                "las": dev_las,
                "uas": dev_uas
            },
            "test": {
                "las": test_las,
                "uas": test_uas
            }
        }

        self.callback_handler.on_eval_end(scores=scores)

        return scores

    def _evaluate_corpus(self, corpus: Corpus, corpus_type: str) -> Tuple[int, int]:
        self.callback_handler.on_epoch_begin(dataset=corpus, corpus_type=corpus_type)

        gold_sentences = []
        pred_sentences = []

        iterator = CorpusIterator(corpus, batch_size=self.batch_size, train=False)
        for i, batch in enumerate(iterator):
            self.callback_handler.on_batch_begin(
                batch=i, batch_data=batch, corpus_type=corpus_type)

            pred, gold = self._run_prediction_batch(batch)
            pred_sentences += pred
            gold_sentences += gold

            self.callback_handler.on_batch_end(
                batch=i, batch_data=batch, gold_sentences=gold, pred_sentences=pred,
                corpus_type=corpus_type)

        las, uas = self.eval_function(gold_sentences, pred_sentences)

        self.callback_handler.on_epoch_end(
            las=las,
            uas=uas,
            gold_sentences=gold_sentences,
            pred_sentences=pred_sentences,
            corpus_type=corpus_type
        )

        return las, uas

    def _run_prediction_batch(self, batch) -> Tuple[List, List]:
        pred_sentences = []
        gold_sentences = []

        sentence_features, sentences = batch

        contextualized_tokens_batch = self.model.get_contextualized_input(
            sentences, sentence_features
        )

        configurations = [
            Configuration(sentence, contextualized_input, self.model,
                          sentence_features=sentence_feature)
            for contextualized_input, sentence, sentence_feature in
            zip(contextualized_tokens_batch, sentences, sentence_features)
        ]

        while configurations:
            # Pass the stacks and buffers through the MLPs in one batch
            configurations = DynamicTrainer.predict_logits(configurations, self.model)

            # The actual computation of the loss must be done sequentially
            for configuration in configurations:
                # Predict a list of possible actions: Transitions, their
                # label (if the transition is LEFT/ RIGHT_ARC) and the
                # score of the action based on the MLP output.
                actions = configuration.predict_actions()

                if not configuration.swap_possible:
                    # Exclude swap options
                    actions = [
                        action for action in actions
                        if action.transition != T.SWAP
                    ]

                assert actions
                best_action = Configuration.get_best_action(actions)

                if best_action.transition == T.SWAP:
                    configuration.num_swap += 1

                configuration.apply_transition(best_action)

                if configuration.is_terminal:
                    pred_sentences.append(configuration.predicted_sentence)
                    gold_sentences.append(configuration.sentence)

            # Remove all finished configurations
            configurations = [c for c in configurations if not c.is_terminal]

        return pred_sentences, gold_sentences
