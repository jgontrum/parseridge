from copy import deepcopy

from parseridge.corpus.treebank import Treebank
from parseridge.parser.configuration import Configuration
from parseridge.parser.attention_model import AttentionModel
from parseridge.utils.helpers import Action, T
from test_parseridge.utils import get_fixtures_path, log_stderr, \
    sentences_are_equal, generate_actions


class TestParseProjective:

    @classmethod
    def setup_class(cls):
        with open(get_fixtures_path("sentence_01.conllu")) as train_io:
            treebank = Treebank(
                train_io=train_io,
                dev_io=None,
                device="cpu"
            )

        cls.corpus = treebank.train_corpus
        cls.vocabulary = treebank.vocabulary
        cls.relations = treebank.relations

        cls.model = AttentionModel(
            relations=cls.relations,
            vocabulary=cls.vocabulary
        )

    @log_stderr
    def test_sentence_parse(self):
        configuration = Configuration(
            sentence=self.corpus.sentences[0],
            contextualized_input=None,
            model=self.model
        )

        # ===============================================================================
        # Iteration 1: SHIFT
        # ===============================================================================

        assert configuration.stack == []
        assert configuration.buffer == [1, 2, 3, 4, 5, 6, 0]
        assert configuration.num_swap == 0
        actions = [
            Action(None, T.SHIFT, 1.0)
        ]

        if not configuration.swap_possible:
            # Exclude swap options
            actions = [
                action for action in actions
                if action.transition != T.SWAP
            ]

        best_action = Configuration.get_best_action(actions)
        if best_action.transition == T.SWAP:
            configuration.num_swap += 1

        assert best_action.transition == T.SHIFT
        assert best_action.relation is None

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [2, 3, 4, 5, 6, 0]
        assert configuration.stack == [1]
        assert not configuration.is_terminal

        # ===============================================================================
        # Iteration 2: SHIFT
        # ===============================================================================

        assert configuration.stack == [1]
        assert configuration.buffer == [2, 3, 4, 5, 6, 0]
        assert configuration.num_swap == 0
        actions = [
          Action(None, T.SHIFT, 1.0), Action(None, T.SWAP, 0.0)
        ] + generate_actions(T.LEFT_ARC, self.relations.relations, [])

        if not configuration.swap_possible:
            # Exclude swap options
            actions = [
                action for action in actions
                if action.transition != T.SWAP
            ]

        best_action = Configuration.get_best_action(actions)
        if best_action.transition == T.SWAP:
            configuration.num_swap += 1

        assert best_action.transition == T.SHIFT
        assert best_action.relation is None

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [3, 4, 5, 6, 0]
        assert configuration.stack == [1, 2]
        assert not configuration.is_terminal

        # ===============================================================================
        # Iteration 3: RIGHT ARC
        # ===============================================================================

        assert configuration.buffer == [3, 4, 5, 6, 0]
        assert configuration.stack == [1, 2]
        assert configuration.num_swap == 0
        actions = [
          Action(None, T.SHIFT, 0.0), Action(None, T.SWAP, 0.0)
        ] + generate_actions(T.LEFT_ARC, self.relations.relations, []) \
          + generate_actions(T.RIGHT_ARC, self.relations.relations, ['punct'])

        if not configuration.swap_possible:
            # Exclude swap options
            actions = [
                action for action in actions
                if action.transition != T.SWAP
            ]

        best_action = Configuration.get_best_action(actions)
        if best_action.transition == T.SWAP:
            configuration.num_swap += 1

        assert best_action.transition == T.RIGHT_ARC
        assert best_action.relation == 'punct'

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [3, 4, 5, 6, 0]
        assert configuration.stack == [1]
        assert not configuration.is_terminal
