import json

from parseridge.corpus.treebank import Treebank
from parseridge.parser.parseridge import ParseRidge
from parseridge.utils.cli_parser import parse_cli_arguments
from parseridge.utils.helpers import set_seed
from parseridge.utils.logger import LoggerMixin


def start():
    logger = LoggerMixin()._logger_setup("Parseridge CLI")
    options = parse_cli_arguments()

    logger.info(
        f"\nHyper Parameters: {json.dumps(vars(options), indent=2, sort_keys=True)}")

    if options.seed:
        set_seed(options.seed)
        logger.warning(f"Set seed to '{options.seed}'."
                       f"This could have a performance impact when run on CUDA.")

    if options.train:
        # Load data
        treebank = Treebank(
            open(options.train_corpus),
            open(options.test_corpus),
            device=options.device
        )

        # Start training
        parser = ParseRidge(options.device)
        parser.fit(
            corpus=treebank.train_corpus,
            relations=treebank.relations,
            dev_corpus=treebank.dev_corpus,
            num_stack=options.num_stack,
            num_buffer=options.num_buffer,
            embedding_size=options.embedding_size,
            lstm_hidden_size=options.lstm_hidden_size,
            lstm_layers=options.lstm_layers,
            relation_mlp_layers=options.relation_mlp_layers,
            transition_mlp_layers=options.transition_mlp_layers,
            margin_threshold=options.margin_threshold,
            error_probability=options.error_probability,
            oov_probability=options.oov_probability,
            token_dropout=options.token_dropout,
            lstm_dropout=options.lstm_dropout,
            mlp_dropout=options.mlp_dropout,
            batch_size=options.batch_size,
            pred_batch_size=options.pred_batch_size,
            num_epochs=options.epochs,
            gradient_clipping=options.gradient_clipping,
            weight_decay=options.weight_decay,
            learning_rate=options.learning_rate,
            update_size=options.update_size,
            loss_factor=options.loss_factor,
            loss_strategy=options.loss_strategy
        )


if __name__ == '__main__':
    start()
