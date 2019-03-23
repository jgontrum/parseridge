import re

from parseridge.corpus.signature import Signature


class Vocabulary(Signature):
    number_regex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")

    def __init__(self):
        super().__init__(entries=["<<<OOV>>>", "<<<PADDING>>>"])

    def _normalize(self, word):
        return 'NUM' if self.number_regex.match(word) else word.lower()

    def add(self, word):
        return super().add(self._normalize(word))

    def get_id(self, word):
        return super().get_id(self._normalize(word))

    def get_count(self, word):
        return super().get_count(self._normalize(word))
