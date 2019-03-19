import numpy as np

from parseridge.corpus.treebank import Treebank
from test_parseridge.utils import log_stderr, get_fixtures_path


@log_stderr
def test_load_treebank():
    with open(get_fixtures_path("sentence_01.conllu")) as train_io:
        treebank = Treebank(
            train_io=train_io,
            dev_io=None,
            device="cpu"
        )

    assert treebank.vocabulary.get_ids() == [
        '<<<OOV>>>', '<<<PADDING>>>', '*root*', 'Aesthetic', 'Appreciation', 'and',
        'Spanish', 'Art', ':'
    ]

    assert treebank.relations.relations == [
        'amod', 'cc', 'conj', 'punct', 'root', 'rroot'
    ]

    train_corpus = treebank.train_corpus
    assert np.array_equal(
        train_corpus.sentence_tensors.numpy(),
        np.array([[[2, 3, 4, 5, 6, 7, 8]]])
    )
