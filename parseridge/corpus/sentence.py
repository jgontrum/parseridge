import random
from copy import deepcopy
from typing import List, Optional, Union, Iterable

import conllu

from parseridge.corpus.token import Token
from parseridge.parser.configuration import Configuration
from parseridge.utils.helpers import Action, T
from parseridge.utils.logger import LoggerMixin


class Sentence(LoggerMixin):
    def __init__(
        self,
        tokens: List[Token],
        text: str = None,
        meta: Optional[dict] = None,
        sentence_id: int = None,
    ):
        self._iter = 0
        self.text = text
        self.meta = meta or {}
        self.id = sentence_id
        self.tokens = [Token.create_root_token()] + tokens

        for token in self:
            if token.head is None:
                token.parent = None
            else:
                token.parent = self.tokens[token.head]

            token.dependents = [
                other_token.id
                for other_token in self.tokens
                if other_token.head == token.id
            ]

        for i, token in enumerate(self._calculate_token_order()):
            token.projective_order = i

        if not self.text:
            self.text = " ".join([token.form for token in self[:-1]])

    def _calculate_token_order(
        self, queue: Optional[List[Token]] = None, index: Optional[int] = None
    ):
        if queue is None:
            queue = [self[0]]
            index = self[0].id
            return self._calculate_token_order(queue, index)
        else:
            results = []

            # Get all the tokens that are dependents of the token
            # at the current index and left to it.
            left_dependents = [token for token in self[:index] if token.head == index]
            for dependent in left_dependents:
                results += self._calculate_token_order(queue, dependent.id)

            # Place the current token in the middle
            results.append(self[index])

            # Get all the dependents right to it
            right_dependents = [token for token in self[index:] if token.head == index]
            for dependent in right_dependents:
                results += self._calculate_token_order(queue, dependent.id)

            return results

    def to_conllu(self) -> conllu.TokenList:
        return conllu.TokenList(
            [token.serialize() for token in self[1:]], metadata=self.meta
        )

    def get_empty_copy(self) -> "Sentence":
        """
        Returns a copy of the sentence but without any gold
        relations or labels. This is used in the training process
        to build a predicted dependency tree from one with
        gold annotations.
        """
        new_tokens = [token.get_unparsed_token() for token in self[1:]]
        return Sentence(new_tokens, text=self.text, meta=self.meta, sentence_id=self.id)

    def __repr__(self) -> str:
        return self.to_conllu().serialize()

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, i: int) -> Union[Token, List[Token]]:
        # Look up tokens for a list of indices
        if isinstance(i, list):
            return [self[j] for j in i]
        # Normal index / slice lookup
        return self.tokens[i]

    def __iter__(self) -> "Sentence":
        return self

    def __next__(self) -> Optional[Token]:
        if self._iter >= len(self):
            self._iter = 0
            raise StopIteration
        else:
            self._iter += 1
            return self[self._iter - 1]

    @classmethod
    def from_conllu(cls, conllu_string: str) -> Iterable["Sentence"]:
        """
        Generator that reads a string containing a treebank in CoNLL-U format
        and produces Sentence objects for all sentences in the treebank.
        :param conllutring:
        :return:
        """
        for sentence in conllu.parse(conllu_string):
            yield cls(
                # Add all tokens, but ignore parataxis (here the id is a tuple)
                tokens=[
                    Token(**token) for token in sentence if isinstance(token["id"], int)
                ],
                text=sentence.metadata["text"],
                meta=sentence.metadata,
            )


class ConfigurationIterator:
    """
    Iterates over a sequence of optimal configurations for this sentence.
    Note that the yielded configuration object is mutable and will change during
    the iteration!
    """

    def __init__(self, sentence):
        self.sentence = deepcopy(sentence)
        self.configuration = Configuration(
            sentence=self.sentence, contextualized_input=None, model=None
        )

    def __next__(self):
        if self.configuration.is_terminal:
            raise StopIteration
        else:
            return self._get_next_configuration(self.configuration)

    def __iter__(self):
        return self

    @staticmethod
    def _get_next_configuration(configuration):
        actions = ConfigurationIterator._get_actions(configuration)
        costs, shift_case = configuration.get_transition_costs(actions)
        valid_actions = [action for action in actions if costs[action.transition] == 0]

        best_action = random.choice(valid_actions)
        configuration.update_dynamic_oracle(best_action, shift_case)
        configuration.apply_transition(best_action)

        return configuration

    @staticmethod
    def _get_actions(configuration):
        actions = []
        if configuration.shift_conditions:
            actions.append(Action(relation=None, transition=T.SHIFT, score=1.0))

        if configuration.swap_conditions:
            actions.append(Action(relation=None, transition=T.SWAP, score=1.0))

        if configuration.left_arc_conditions:
            actions.append(
                Action(
                    relation=configuration.top_stack_token.relation,
                    transition=T.LEFT_ARC,
                    score=1.0,
                )
            )

        if configuration.right_arc_conditions:
            actions.append(
                Action(
                    relation=configuration.top_stack_token.relation,
                    transition=T.RIGHT_ARC,
                    score=1.0,
                )
            )

        return actions
