import re
from typing import Optional, List, Set

from parseridge.corpus.signature import Signature


class Vocabulary(Signature[str]):
    number_regex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")

    def __init__(
        self,
        entries: Optional[List[str]] = None,
        embeddings_vocab: Optional[Set[str]] = None,
    ):
        default_entries = ["<<<OOV>>>", "<<<PADDING>>>", "NUM", "*root*"]
        self.embeddings_vocab = None  # Init with None to be able to add default_entries

        if entries is not None:
            default_entries += entries
        super().__init__(entries=default_entries)

        self.oov = self.get_id("<<<OOV>>>")
        self.embeddings_vocab = embeddings_vocab

    def _normalize(self, word):
        return "NUM" if self.number_regex.match(word) else word.lower()

    def add(self, word):
        word = self._normalize(word)

        if self.embeddings_vocab and word not in self.embeddings_vocab:
            # Set this as OOV, as it is not present in the embeddings.
            return self.oov

        return super().add(self._normalize(word))

    def get_id(self, word):
        return super().get_id(self._normalize(word))

    def get_count(self, word):
        return super().get_count(self._normalize(word))
