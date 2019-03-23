from parseridge.corpus.corpus import Corpus
from parseridge.corpus.relations import Relations
from parseridge.corpus.sentence import Sentence
from parseridge.corpus.signature import Signature
from parseridge.corpus.vocabulary import Vocabulary
from parseridge.utils.logger import LoggerMixin


class Treebank(LoggerMixin):
    """
    Wrapper for multiple corpora and handling of sharing signatures for
    tokens, labels etc. between them.
    """

    def __init__(self, train_io, dev_io, test_io=None, device="cpu"):
        train_as_string = "".join(train_io)
        dev_as_string = "".join(dev_io)
        test_as_string = "".join(test_io) if test_io else ""

        self.logger.info(f"Load training corpus...")
        train_sentences = list(Sentence.from_conllu(train_as_string))

        self.vocabulary = Vocabulary()

        self.relations = Relations(train_sentences)

        self.train_corpus = Corpus(
            sentences=train_sentences,
            vocabulary=self.vocabulary,
            device=device
        )

        self.vocabulary.read_only()

        self.logger.info(f"Load development corpus...")

        self.dev_corpus = Corpus(
            sentences=list(Sentence.from_conllu(dev_as_string)),
            vocabulary=self.vocabulary,
            device=device
        )

        self.test_corpus = None
        if test_as_string:
            self.logger.info(f"Load test corpus...")
            self.test_corpus = Corpus(
                sentences=list(Sentence.from_conllu(test_as_string)),
                vocabulary=self.vocabulary,
                device=device
            )
