import json
import logging

from torch import nn
from torch.optim import Adam

from parseridge import formatter, logger
from parseridge.corpus.treebank import Treebank
from parseridge.corpus.vocabulary import Vocabulary
from parseridge.parser.activation import ACTIVATION_FUNCTIONS
from parseridge.parser.attention_model import AttentionModel
from parseridge.parser.evaluation import Evaluator
from parseridge.parser.evaluation.callbacks import EvalProgressBarCallback, EvalSimpleLogger
from parseridge.parser.evaluation.callbacks.csv_callback import CSVReporter
from parseridge.parser.evaluation.callbacks.google_sheets_callback import (
    GoogleSheetsReporter,
)
from parseridge.parser.modules.external_embeddings import ExternalEmbeddings
from parseridge.parser.training.callbacks.evaluation_callback import EvaluationCallback
from parseridge.parser.training.callbacks.gradient_clipping_callback import (
    GradientClippingCallback,
)
from parseridge.parser.training.callbacks.partial_freeze_embeddings_callback import (
    PartialFreezeEmbeddingsCallback,
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

    logger.info(f"Hyper Parameters: \n{json.dumps(vars(args), indent=2, sort_keys=True)}")

    try:
        # Set the seed for deterministic outcomes
        if args.seed:
            logger.info(f"Setting random seed to {args.seed}.")
            set_seed(args.seed)

        # Load external embeddings
        if args.embeddings_file:
            logger.info(f"Loading embeddings from '{args.embeddings_file}'...")
            embeddings = ExternalEmbeddings(
                path=args.embeddings_file, vendor=args.embeddings_vendor
            )

            vocabulary = Vocabulary(embeddings_vocab=embeddings.vocab)

            if embeddings.dim != args.embedding_size:
                logger.warning(
                    f"Embedding dimension mismatch: External embeddings have "
                    f"{embeddings.dim} dimensions, settings require "
                    f"{args.embedding_size} dimensions. Overwriting settings..."
                )
                args.embedding_size = embeddings.dim
        else:
            embeddings = None
            vocabulary = None

        # Load the corpora
        treebank = Treebank(
            train_io=open(args.train_corpus),
            dev_io=open(args.dev_corpus),
            test_io=open(args.test_corpus),
            vocabulary=vocabulary,
            device=args.device,
        )

        # Configure the machine learning model
        model = AttentionModel(
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
            embeddings=embeddings,
            transition_mlp_activation=ACTIVATION_FUNCTIONS[args.transition_mlp_activation],
            relation_mlp_activation=ACTIVATION_FUNCTIONS[args.relation_mlp_activation],
            self_attention_heads=args.self_attention_heads,
            scale_query=args.scale_query,
            scale_key=args.scale_key,
            scale_value=args.scale_value,
            scoring_function=args.scoring_function,
            normalization_function=args.normalization_function,
            device=args.device,
        ).to(args.device)

        optimizer = Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        # Set-up callbacks for the training and the evaluation.
        evaluation_callbacks = [
            EvalSimpleLogger(),
            CSVReporter(csv_path=args.csv_output_file),
            GoogleSheetsReporter(
                experiment_title=args.experiment_name,
                sheets_id=args.google_sheets_id,
                auth_file_path=args.google_sheets_auth_path,
                hyper_parameters=vars(args),
            ),
        ]

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

        if embeddings and args.freeze_embeddings:
            training_callbacks.append(
                PartialFreezeEmbeddingsCallback(
                    freeze_indices=embeddings.freeze_indices.clone(),
                    embedding_layer=model.input_encoder.token_embeddings,
                )
            )
            del embeddings

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
    except Exception:
        # If the trainer crashes, we want to save the exception into our logs.
        logger.error("Fatal error in main loop:", exc_info=True)
