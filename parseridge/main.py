import logging
import torch

import numpy as np

import os
from parseridge.corpus.corpus import Corpus
from parseridge.corpus.sentence import Sentence
from parseridge.corpus.signature import Signature

from parseridge.corpus.treebank import Treebank
from parseridge.parser.parseridge import ParseRidge
from parseridge.utils.cli_parser import parse_cli_arguments
from parseridge.utils.evaluate import CoNNLEvaluator
from parseridge.utils.helpers import get_device, set_seed

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def start():
    device = get_device()
    options = parse_cli_arguments()

    if options.seed:
        set_seed(options.seed)
        logger.warning(f"Set seed to '{options.seed}'."
                       f"This could have a performance impact when run on CUDA.")

    # Load data
    treebank = Treebank(
        open(options.train),
        open(options.test),
        device=device
    )

    # Start training
    parser = ParseRidge(device)
    parser.fit(
        treebank.train_corpus,
        batch_size=options.batch_size,
        error_prob=options.error_probability,
        dropout=options.dropout,
        relations=treebank.relations,
        num_epochs=30,
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
