import conllu

from parseridge.corpus.token import Token
from parseridge.utils.logger import LoggerMixin


class Sentence(LoggerMixin):

    def __init__(self, tokens, text=None, meta=None, sentence_id=None):
        self._iter = 0
        self.text = text
        if not meta:
            meta = {}
        self.meta = meta
        self.id = sentence_id
        self.tokens = [Token.create_root_token()] + tokens

        for token in self:
            token.parent = None if token.head is None \
                else self.tokens[token.head]
            token.dependents = [
                other_token.id for other_token in self.tokens
                if other_token.head == token.id
            ]

        for i, token in enumerate(self._calculate_token_order()):
            token.projective_order = i

        if not self.text:
            self.text = " ".join([token.form for token in self[:-1]])

    def _calculate_token_order(self, queue=None, index=None):
        if queue is None:
            queue = [self[0]]
            index = self[0].id
            return self._calculate_token_order(queue, index)
        else:
            results = []

            # Get all the tokens that are dependents of the token
            # at the current index and left to it.
            left_dependents = [
                token for token in self[:index] if token.head == index
            ]
            for dependent in left_dependents:
                results += self._calculate_token_order(queue, dependent.id)

            # Place the current token in the middle
            results.append(self[index])

            # Get all the dependents right to it
            right_dependents = [
                token for token in self[index:] if token.head == index
            ]
            for dependent in right_dependents:
                results += self._calculate_token_order(queue, dependent.id)

            return results

    def to_conllu(self):
        return conllu.TokenList(
            [token.serialize() for token in self[1:]],
            metadata=self.meta
        )

    def get_empty_copy(self):
        """
        Returns a copy of the sentence but without any gold
        relations or labels. This is used in the training process
        to build a predicted dependency tree from one with
        gold annotations.
        """
        new_tokens = [token.get_unparsed_token() for token in self[1:]]
        return Sentence(new_tokens, text=self.text, meta=self.meta, sentence_id=self.id)

    def __repr__(self):
        return self.to_conllu().serialize()

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        # Look up tokens for a list of indices
        if isinstance(i, list):
            return [
                self[j] for j in i
            ]
        # Normal index / slice lookup
        return self.tokens[i]

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter >= len(self):
            self._iter = 0
            raise StopIteration
        else:
            self._iter += 1
            return self[self._iter - 1]

    @classmethod
    def from_conllu(cls, conllu_string):
        """
        Generator that reads a string containing a treebank in CoNLL-U format
        and produces Sentence objects for all sentences in the treebank.
        :param conllutring:
        :return:
        """
        for sentence in conllu.parse(conllu_string):
            yield cls(
                # Add all tokens, but ignore parataxis (here the id is a tuple)
                tokens=[Token(**token) for token in sentence
                        if isinstance(token["id"], int)],
                text=sentence.metadata["text"],
                meta=sentence.metadata
            )
