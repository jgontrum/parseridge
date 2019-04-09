from copy import deepcopy

from parseridge.corpus.treebank import Treebank
from parseridge.parser.configuration import Configuration
from parseridge.parser.model import ParseridgeModel
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

        cls.model = ParseridgeModel(
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
        actions = [
            Action(None, T.SHIFT, 1.0)
        ]

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {
            T.LEFT_ARC: 1,
            T.RIGHT_ARC: 1,
            T.SHIFT: 0,
            T.SWAP: 1
        }
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = \
            configuration.select_actions(actions, costs, error_probability=0.0)

        assert best_action.transition == T.SHIFT
        assert best_wrong_action == Action.get_negative_action()

        # Update oracle
        sentence_before = deepcopy(configuration.sentence)
        configuration.update_dynamic_oracle(best_action, shift_case)
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [2, 3, 4, 5, 6, 0]
        assert configuration.stack == [1]
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # ===============================================================================
        # Iteration 2: LEFT ARC
        # ===============================================================================

        assert configuration.buffer == [2, 3, 4, 5, 6, 0]
        assert configuration.stack == [1]
        actions = [
            Action(None, T.SHIFT, 0.0), Action(None, T.SWAP, 0.0)
        ] + generate_actions(T.LEFT_ARC, self.relations.relations, ['amod', 'punct'])

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {
            T.LEFT_ARC: 0,
            T.RIGHT_ARC: 1,
            T.SHIFT: 1,
            T.SWAP: 1
        }
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = \
            configuration.select_actions(actions, costs, error_probability=0.0)

        assert best_action.transition == T.LEFT_ARC
        assert best_action.relation == 'amod'
        assert best_wrong_action.transition == T.LEFT_ARC
        assert best_wrong_action.relation == 'punct'

        # Update oracle
        configuration.update_dynamic_oracle(best_action, shift_case)
        top_stack = configuration.top_stack_token
        assert top_stack.dependents == []
        assert top_stack.id not in top_stack.parent.dependents

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [2, 3, 4, 5, 6, 0]
        assert configuration.stack == []
        assert configuration.predicted_sentence[1].head == 2
        assert configuration.predicted_sentence[1].relation == 'amod'

        # ===============================================================================
        # Iteration 3: SHIFT
        # ===============================================================================

        assert configuration.stack == []
        assert configuration.buffer == [2, 3, 4, 5, 6, 0]
        actions = [
            Action(None, T.SHIFT, 1.0)
        ]

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {
            T.LEFT_ARC: 1,
            T.RIGHT_ARC: 1,
            T.SHIFT: 0,
            T.SWAP: 1
        }
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = \
            configuration.select_actions(actions, costs, error_probability=0.0)

        assert best_action.transition == T.SHIFT
        assert best_wrong_action == Action.get_negative_action()

        # Update oracle
        sentence_before = deepcopy(configuration.sentence)
        configuration.update_dynamic_oracle(best_action, shift_case)
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [3, 4, 5, 6, 0]
        assert configuration.stack == [2]
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # ===============================================================================
        # Iteration 4: SHIFT
        # ===============================================================================

        assert configuration.buffer == [3, 4, 5, 6, 0]
        assert configuration.stack == [2]
        actions = [
          Action(None, T.SHIFT, 1.0), Action(None, T.SWAP, 0.0)
        ] + generate_actions(T.LEFT_ARC, self.relations.relations, ['punct'])

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {
            T.LEFT_ARC: 3,
            T.RIGHT_ARC: 1,
            T.SHIFT: 0,
            T.SWAP: 1
        }
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = \
            configuration.select_actions(actions, costs, error_probability=0.0)

        assert best_action.transition == T.SHIFT
        assert best_wrong_action.transition == T.LEFT_ARC
        assert best_wrong_action.relation == 'punct'

        # Update oracle
        sentence_before = deepcopy(configuration.sentence)
        configuration.update_dynamic_oracle(best_action, shift_case)
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [4, 5, 6, 0]
        assert configuration.stack == [2, 3]
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # ===============================================================================
        # Iteration 5: SHIFT
        # ===============================================================================

        assert configuration.buffer == [4, 5, 6, 0]
        assert configuration.stack == [2, 3]
        actions = [
            Action(None, T.SHIFT, 1.0), Action(None, T.SWAP, 0.0)
          ] + generate_actions(T.LEFT_ARC, self.relations.relations, []) \
            + generate_actions(T.RIGHT_ARC, self.relations.relations, ['rroot'])

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {
            T.LEFT_ARC: 1,
            T.RIGHT_ARC: 1,
            T.SHIFT: 0,
            T.SWAP: 1
        }
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = \
            configuration.select_actions(actions, costs, error_probability=0.0)

        assert best_action.transition == T.SHIFT
        assert best_wrong_action.transition == T.RIGHT_ARC
        assert best_wrong_action.relation == 'rroot'

        # Update oracle
        sentence_before = deepcopy(configuration.sentence)
        configuration.update_dynamic_oracle(best_action, shift_case)
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [5, 6, 0]
        assert configuration.stack == [2, 3, 4]
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # ===============================================================================
        # Iteration 6: LEFT ARC
        # ===============================================================================

        assert configuration.buffer == [5, 6, 0]
        assert configuration.stack == [2, 3, 4]
        actions = [
            Action(None, T.SHIFT, 0.0), Action(None, T.SWAP, 0.0)
          ] + generate_actions(T.LEFT_ARC, self.relations.relations, ['amod']) \
            + generate_actions(T.RIGHT_ARC, self.relations.relations, ['rroot'])

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {
            T.LEFT_ARC: 0,
            T.RIGHT_ARC: 1,
            T.SHIFT: 3,
            T.SWAP: 1
        }
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = \
            configuration.select_actions(actions, costs, error_probability=0.0)

        assert best_action.transition == T.LEFT_ARC
        assert best_action.relation == 'amod'
        assert best_wrong_action.transition == T.RIGHT_ARC
        assert best_wrong_action.relation == 'rroot'

        # Update oracle
        configuration.update_dynamic_oracle(best_action, shift_case)
        top_stack = configuration.top_stack_token
        assert top_stack.dependents == []
        assert top_stack.id not in top_stack.parent.dependents

        # Apply transition
        configuration.apply_transition(best_action)
        assert configuration.buffer == [5, 6, 0]
        assert configuration.stack == [2, 3]
        assert configuration.predicted_sentence[4].head == 5
        assert configuration.predicted_sentence[4].relation == 'amod'

        # ===============================================================================
        # Iteration 7: LEFT ARC
        # ===============================================================================

        assert configuration.buffer == [5, 6, 0]
        assert configuration.stack == [2, 3]
        actions = [
            Action(None, T.SHIFT, 0.0), Action(None, T.SWAP, 0.0)
          ] + generate_actions(T.LEFT_ARC, self.relations.relations, ['cc']) \
            + generate_actions(T.RIGHT_ARC, self.relations.relations, ['conj'])

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {
            T.LEFT_ARC: 0,
            T.RIGHT_ARC: 1,
            T.SHIFT: 2,
            T.SWAP: 1
        }
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = \
            configuration.select_actions(actions, costs, error_probability=0.0)

        assert best_action.transition == T.LEFT_ARC
        assert best_action.relation == 'cc'
        assert best_wrong_action.transition == T.RIGHT_ARC
        assert best_wrong_action.relation == 'conj'

        # Update oracle
        configuration.update_dynamic_oracle(best_action, shift_case)
        top_stack = configuration.top_stack_token
        assert top_stack.dependents == []
        assert top_stack.id not in top_stack.parent.dependents

        # Apply transition
        configuration.apply_transition(best_action)
        assert configuration.buffer == [5, 6, 0]
        assert configuration.stack == [2]
        assert configuration.predicted_sentence[3].head == 5
        assert configuration.predicted_sentence[3].relation == 'cc'

        # ===============================================================================
        # Iteration 8: SHIFT
        # ===============================================================================

        assert configuration.buffer == [5, 6, 0]
        assert configuration.stack == [2]
        actions = [
            Action(None, T.SHIFT, 1.0), Action(None, T.SWAP, 1.0)
          ] + generate_actions(T.LEFT_ARC, self.relations.relations, [])

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {
            T.LEFT_ARC: 3,
            T.RIGHT_ARC: 1,
            T.SHIFT: 0,
            T.SWAP: 1
        }
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = \
            configuration.select_actions(actions, costs, error_probability=0.0)

        assert best_action.transition == T.SHIFT
        assert best_wrong_action.transition == T.SWAP

        # Update oracle
        sentence_before = deepcopy(configuration.sentence)
        configuration.update_dynamic_oracle(best_action, shift_case)
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [6, 0]
        assert configuration.stack == [2, 5]
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # ===============================================================================
        # Iteration 9: RIGHT ARC
        # ===============================================================================

        assert configuration.buffer == [6, 0]
        assert configuration.stack == [2, 5]
        actions = [
            Action(None, T.SHIFT, 0.0), Action(None, T.SWAP, 0.0)
          ] + generate_actions(T.LEFT_ARC, self.relations.relations, []) \
            + generate_actions(T.RIGHT_ARC, self.relations.relations, ['conj', 'rroot'])

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {
            T.LEFT_ARC: 1,
            T.RIGHT_ARC: 0,
            T.SHIFT: 1,
            T.SWAP: 1
        }
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = \
            configuration.select_actions(actions, costs, error_probability=0.0)

        assert best_action.transition == T.RIGHT_ARC
        assert best_action.relation == 'conj'
        assert best_wrong_action.transition == T.RIGHT_ARC
        assert best_wrong_action.relation == 'rroot'

        # Update oracle
        configuration.update_dynamic_oracle(best_action, shift_case)
        top_stack = configuration.top_stack_token
        assert top_stack.dependents == []
        assert top_stack.id not in top_stack.parent.dependents

        # Apply transition
        configuration.apply_transition(best_action)
        assert configuration.buffer == [6, 0]
        assert configuration.stack == [2]
        assert configuration.predicted_sentence[5].head == 2
        assert configuration.predicted_sentence[5].relation == 'conj'

        # ===============================================================================
        # Iteration 10: SHIFT
        # ===============================================================================

        assert configuration.buffer == [6, 0]
        assert configuration.stack == [2]
        actions = [
            Action(None, T.SHIFT, 1.0), Action(None, T.SWAP, 1.0)
        ] + generate_actions(T.LEFT_ARC, self.relations.relations, [])

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {
            T.LEFT_ARC: 2,
            T.RIGHT_ARC: 1,
            T.SHIFT: 0,
            T.SWAP: 1
        }
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = \
            configuration.select_actions(actions, costs, error_probability=0.0)

        assert best_action.transition == T.SHIFT
        assert best_wrong_action.transition == T.SWAP

        # Update oracle
        sentence_before = deepcopy(configuration.sentence)
        configuration.update_dynamic_oracle(best_action, shift_case)
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [0]
        assert configuration.stack == [2, 6]
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # ===============================================================================
        # Iteration 11: RIGHT ARC
        # ===============================================================================

        assert configuration.buffer == [0]
        assert configuration.stack == [2, 6]
        actions = generate_actions(T.LEFT_ARC, self.relations.relations, []) \
             + generate_actions(T.RIGHT_ARC, self.relations.relations, ['punct', 'rroot'])

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {
            T.LEFT_ARC: 1,
            T.RIGHT_ARC: 0,
            T.SHIFT: 1,
            T.SWAP: 1
        }
        assert shift_case == 0

        # Get best actions
        best_action, best_valid_action, best_wrong_action = \
            configuration.select_actions(actions, costs, error_probability=0.0)

        assert best_action.transition == T.RIGHT_ARC
        assert best_action.relation == 'punct'
        assert best_wrong_action.transition == T.RIGHT_ARC
        assert best_wrong_action.relation == 'rroot'

        # Update oracle
        configuration.update_dynamic_oracle(best_action, shift_case)
        top_stack = configuration.top_stack_token
        assert top_stack.dependents == []
        assert top_stack.id not in top_stack.parent.dependents

        # Apply transition
        configuration.apply_transition(best_action)
        assert configuration.buffer == [0]
        assert configuration.stack == [2]
        assert configuration.predicted_sentence[6].head == 2
        assert configuration.predicted_sentence[6].relation == 'punct'

        # ===============================================================================
        # Iteration 7: LEFT ARC
        # ===============================================================================

        assert configuration.buffer == [0]
        assert configuration.stack == [2]
        actions = generate_actions(T.LEFT_ARC, self.relations.relations, ['root', 'cc'])

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {
            T.LEFT_ARC: 0,
            T.RIGHT_ARC: 1,
            T.SHIFT: 1,
            T.SWAP: 1
        }
        assert shift_case == 0

        # Get best actions
        best_action, best_valid_action, best_wrong_action = \
            configuration.select_actions(actions, costs, error_probability=0.0)

        assert best_action.transition == T.LEFT_ARC
        assert best_action.relation == 'root'
        assert best_wrong_action.transition == T.LEFT_ARC
        assert best_wrong_action.relation == 'cc'

        # Update oracle
        configuration.update_dynamic_oracle(best_action, shift_case)
        top_stack = configuration.top_stack_token
        assert top_stack.dependents == []
        assert top_stack.id not in top_stack.parent.dependents

        # Apply transition
        configuration.apply_transition(best_action)
        assert configuration.buffer == [0]
        assert configuration.stack == []
        assert configuration.predicted_sentence[2].head == 0
        assert configuration.predicted_sentence[2].relation == 'root'
