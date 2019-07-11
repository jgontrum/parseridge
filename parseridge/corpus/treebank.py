from dataclasses import dataclass
from typing import Optional, List, TextIO

from parseridge.corpus.corpus import Corpus
from parseridge.corpus.relations import Relations
from parseridge.corpus.sentence import Sentence
from parseridge.corpus.vocabulary import Vocabulary
from parseridge.utils.logger import LoggerMixin


@dataclass
class Treebank(LoggerMixin):
    """
    Wrapper for multiple corpora and handling of sharing signatures for
    tokens, labels etc. between them.
    """

    train_io: Optional[TextIO] = None
    dev_io: Optional[TextIO] = None
    test_io: Optional[TextIO] = None

    train_sentences: Optional[List[Sentence]] = None
    dev_sentences: Optional[List[Sentence]] = None
    test_sentences: Optional[List[Sentence]] = None

    vocabulary: Optional[Vocabulary] = None
    relations: Optional[Relations] = None

    device: str = "cpu"

    def __post_init__(self):
        if not self.train_sentences:
            self.logger.info("Load training corpus...")
            as_string = "".join(self.train_io)
            self.train_sentences = list(Sentence.from_conllu(as_string))

        if not self.dev_sentences and self.dev_io:
            self.logger.info("Load development corpus...")
            as_string = "".join(self.dev_io)
            self.dev_sentences = list(Sentence.from_conllu(as_string))

        if not self.test_sentences and self.test_io:
            self.logger.info("Load test corpus...")
            as_string = "".join(self.test_io)
            self.test_sentences = list(Sentence.from_conllu(as_string))

        if not self.vocabulary:
            # Mode 1
            self.vocabulary = Vocabulary()
            self.relations = Relations(self.train_sentences)
        else:
            # Mode 2
            self.vocabulary.read_only()

        self.logger.info("Creating corpus objects...")
        self.train_corpus = Corpus(
            sentences=self.train_sentences, vocabulary=self.vocabulary, device=self.device
        )

        self.vocabulary.read_only()

        self.dev_corpus = (
            Corpus(
                sentences=self.dev_sentences, vocabulary=self.vocabulary, device=self.device
            )
            if self.dev_sentences
            else None
        )

        self.test_corpus = (
            Corpus(
                sentences=self.test_sentences,
                vocabulary=self.vocabulary,
                device=self.device,
            )
            if self.test_sentences
            else None
        )
