from torch.optim import Adam

from parseridge import logger, git_commit
from parseridge.parser.evaluation.callbacks.attention_reporter_callback import (
    AttentionReporter,
)
from parseridge.parser.model import ParseridgeModel
from parseridge.parser.evaluation import Evaluator
from parseridge.parser.evaluation.callbacks import EvalProgressBarCallback, EvalSimpleLogger
from parseridge.parser.training.callbacks.evaluation_callback import EvaluationCallback
from parseridge.parser.training.callbacks.progress_bar_callback import ProgressBarCallback
from parseridge.parser.training.dynamic_trainer import DynamicTrainer

from parseridge.corpus.treebank import Treebank

logger.info(f"Running at git commit {git_commit}.")

treebank = Treebank(
    train_io=open("data/UD_English-GUM/en_gum-ud-train.conllu"),
    dev_io=open("data/UD_English-GUM/en_gum-ud-dev.conllu"),
)

attention_reporter = AttentionReporter(
    file_path="./attention_energies/", vocabulary=treebank.vocabulary
)

model = ParseridgeModel(
    relations=treebank.relations,
    vocabulary=treebank.vocabulary,
    configuration_encoder="universal_attention",
    input_encoder_type="transformer",
    self_attention_heads=5,
    self_attention_layers=2,
    # attention_reporter=attention_reporter,
    # scale_key=125,
    # scale_value=125,
    # scale_query=125,
    scoring_function="concat",
)

optimizer = Adam(model.parameters())

evaluator = Evaluator(
    model,
    treebank,
    callbacks=[EvalProgressBarCallback(), EvalSimpleLogger(), attention_reporter],
)

callbacks = [ProgressBarCallback(moving_average=64), EvaluationCallback(evaluator)]

trainer = DynamicTrainer(model, optimizer, callbacks=callbacks)
trainer.fit(epochs=2, training_data=treebank.train_corpus, batch_size=4)
