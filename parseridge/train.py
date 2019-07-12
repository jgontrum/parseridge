import logging

from torch import nn
from torch.optim import Adam

from parseridge import formatter, logger
from parseridge.corpus.treebank import Treebank
from parseridge.parser.baseline_model import BaselineModel
from parseridge.parser.evaluation import Evaluator
from parseridge.parser.evaluation.callbacks import EvalProgressBarCallback, EvalSimpleLogger
from parseridge.parser.evaluation.callbacks.csv_callback import CSVReporter
from parseridge.parser.modules.external_embeddings import ExternalEmbeddings
from parseridge.parser.training.callbacks.evaluation_callback import EvaluationCallback
from parseridge.parser.training.callbacks.gradient_clipping_callback import (
    GradientClippingCallback,
)
from parseridge.parser.training.callbacks.progress_bar_callback import ProgressBarCallback
from parseridge.parser.training.callbacks.save_model_callback import SaveModelCallback
from parseridge.parser.training.callbacks.simple_logger_callback import (
    TrainSimpleLoggerCallback,
)
from parseridge.parser.training.dynamic_trainer import DynamicTrainer
from parseridge.utils.cli_parser import parse_train_cli_arguments
from parseridge.utils.helpers import set_seed


if __name__ == "__main__":
    args = parse_train_cli_arguments()

    # Configure logger to save to file
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Set the seed for deterministic outcomes
    if args.seed:
        set_seed(args.seed)

    # Load the corpora
    treebank = Treebank(
        train_io=open("data/UD_English-GUM/en_gum-ud-train.conllu"),
        dev_io=open("data/UD_English-GUM/en_gum-ud-dev.conllu"),
        test_io=None,
        device=args.device,
    )

    # Load external embeddings
    if args.embeddings_file:
        logger.info(f"Loading embeddings from '{args.embeddings_file}'...")
        embeddings = ExternalEmbeddings(
            path=args.embeddings_file,
            vendor=args.embeddings_vendor,
            freeze=args.freeze_embeddings,
        )
    else:
        embeddings = None

    # Configure the machine learning model
    model = BaselineModel(
        relations=treebank.relations,
        vocabulary=treebank.vocabulary,
        num_stack=args.num_stack,
        num_buffer=args.num_buffer,
        lstm_dropout=args.lstm_dropout,
        mlp_dropout=args.mlp_dropout,
        embedding_size=args.embedding_size,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        input_encoder_type=args.input_encoder_type,
        relation_mlp_layers=args.relation_mlp_layers,
        transition_mlp_layers=args.transition_mlp_layers,
        device=args.device,
    ).to(args.device)

    optimizer = Adam(model.parameters(), lr=0, weight_decay=0)

    # Set-up callbacks for the training and the evaluation.
    evaluation_callbacks = [EvalSimpleLogger(), CSVReporter(csv_path=args.csv_output_file)]

    training_callbacks = [
        TrainSimpleLoggerCallback(),
        SaveModelCallback(folder_path=args.model_save_path),
    ]

    if args.show_progress_bars:
        evaluation_callbacks.append(EvalProgressBarCallback())
        training_callbacks.append(ProgressBarCallback(moving_average=64))

    training_callbacks.append(
        EvaluationCallback(
            evaluator=Evaluator(model, treebank, callbacks=evaluation_callbacks)
        )
    )

    # Enable gradient clipping
    if args.gradient_clipping:
        training_callbacks.append(
            GradientClippingCallback(threshold=args.gradient_clipping)
        )

    # Get loss function
    loss_function = {
        "MaxMargin": None,  # The default, in-built loss function,
        "CrossEntropy": nn.CrossEntropyLoss(),
    }[args.loss_function]

    # Create the trainer and start training
    trainer = DynamicTrainer(model, optimizer, callbacks=training_callbacks)
    trainer.fit(
        epochs=args.epochs,
        training_data=treebank.train_corpus,
        batch_size=args.batch_size,
        error_probability=args.error_probability,
        oov_probability=args.oov_probability,
        token_dropout=args.token_dropout,
        margin_threshold=args.margin_threshold,
        update_frequency=args.update_frequency,
        criterion=loss_function,
    )