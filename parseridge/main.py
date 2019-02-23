import logging

import numpy as np
import torch

from parseridge.corpus.treebank import Treebank
from parseridge.parser.parseridge import ParseRidge
from parseridge.utils.cli_parser import parse_cli_arguments

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def start():
    options = parse_cli_arguments()

    # Set seed
    seed = options.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load data
    treebank = Treebank(
        open(options.train),
        open(options.test),
        device=options.device
    )

    # Start training
    parser = ParseRidge(options.device)
    parser.fit(
        treebank.train_corpus,
        batch_size=options.batch_size,
        error_prob=options.error_probability,
        dropout=options.dropout,
        relations=treebank.relations,
        num_epochs=60,
        dev_corpus=treebank.dev_corpus
    )

    # parser = ParseRidge(device)
    # parser.load_model("models/model-20190214-121803.model")
    # parser.model.vocabulary.read_only()
    #
    # dev_as_string = "".join(open(options.test))
    # train_sentences = list(Sentence.from_conllu(dev_as_string))
    # #
    # corpus = Corpus(
    #     sentences=train_sentences,
    #     vocabulary=Signature(entries=["<<<PADDING>>>", "<<<OOV>>>"]),
    #     device=device
    # )
    #
    # for feats, sents in corpus.get_iterator(128, True, False):
    #     print(len(sents))

    #
    # pred, gold = parser.predict(corpus)
    #
    # evaluator = CoNNLEvaluator()
    # print(evaluator.get_las_score_for_sentences(pred, gold))


if __name__ == '__main__':
    start()
