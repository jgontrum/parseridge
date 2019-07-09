from copy import deepcopy

from parseridge.corpus.treebank import Treebank
from parseridge.parser.configuration import Configuration
from parseridge.parser.attention_model import AttentionModel
from parseridge.utils.helpers import Action, T
from test_parseridge.utils import (
    get_fixtures_path,
    log_stderr,
    sentences_are_equal,
    generate_actions,
)


class TestParseNonProjectiveSentence:
    @classmethod
    def setup_class(cls):
        with open(get_fixtures_path("sentence_02.conllu")) as train_io:
            treebank = Treebank(train_io=train_io, dev_io=None, device="cpu")

        cls.corpus = treebank.train_corpus
        cls.vocabulary = treebank.vocabulary
        cls.relations = treebank.relations

        cls.model = AttentionModel(relations=cls.relations, vocabulary=cls.vocabulary)

    @log_stderr
    def test_sentence_parse(self):
        configuration = Configuration(
            sentence=self.corpus.sentences[0], contextualized_input=None, model=self.model
        )

        # ===============================================================================
        # Iteration 1: SHIFT
        # ===============================================================================

        assert configuration.stack == []
        assert configuration.buffer == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        actions = [Action(None, T.SHIFT, 1.0)]

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {T.LEFT_ARC: 1, T.RIGHT_ARC: 1, T.SHIFT: 0, T.SWAP: 1}
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = configuration.select_actions(
            actions, costs, error_probability=0.0
        )

        assert best_action.transition == T.SHIFT
        assert best_wrong_action == Action.get_negative_action()

        # Update oracle
        sentence_before = deepcopy(configuration.sentence)
        configuration.update_dynamic_oracle(best_action, shift_case)
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [2, 3, 4, 5, 6, 7, 8, 9, 0]
        assert configuration.stack == [1]
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # ===============================================================================
        # Iteration 2: SHIFT
        # ===============================================================================

        assert configuration.buffer == [2, 3, 4, 5, 6, 7, 8, 9, 0]
        assert configuration.stack == [1]
        actions = [
            Action(None, T.SHIFT, 1.0),
            Action(None, T.SWAP, 1.0),
        ] + generate_actions(T.LEFT_ARC, self.relations.relations, [])

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {T.LEFT_ARC: 1, T.RIGHT_ARC: 1, T.SHIFT: 0, T.SWAP: 1}
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = configuration.select_actions(
            actions, costs, error_probability=0.0
        )

        assert best_action.transition == T.SHIFT
        assert best_wrong_action.transition == T.SWAP

        # Update oracle
        sentence_before = deepcopy(configuration.sentence)
        configuration.update_dynamic_oracle(best_action, shift_case)
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [3, 4, 5, 6, 7, 8, 9, 0]
        assert configuration.stack == [1, 2]
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # ===============================================================================
        # Iteration 3: SHIFT
        # ===============================================================================

        assert configuration.buffer == [3, 4, 5, 6, 7, 8, 9, 0]
        assert configuration.stack == [1, 2]
        actions = (
            [Action(None, T.SHIFT, 1.0), Action(None, T.SWAP, 0.0)]
            + generate_actions(T.LEFT_ARC, self.relations.relations, ["rroot"])
            + generate_actions(T.RIGHT_ARC, self.relations.relations, [])
        )

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {T.LEFT_ARC: 1, T.RIGHT_ARC: 1, T.SHIFT: 0, T.SWAP: 1}
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = configuration.select_actions(
            actions, costs, error_probability=0.0
        )

        assert best_action.transition == T.SHIFT
        assert best_wrong_action.transition == T.LEFT_ARC
        assert best_wrong_action.relation == "rroot"

        # Update oracle
        sentence_before = deepcopy(configuration.sentence)
        configuration.update_dynamic_oracle(best_action, shift_case)
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [4, 5, 6, 7, 8, 9, 0]
        assert configuration.stack == [1, 2, 3]
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # ===============================================================================
        # Iteration 4: LEFT ARC
        # ===============================================================================

        assert configuration.buffer == [4, 5, 6, 7, 8, 9, 0]
        assert configuration.stack == [1, 2, 3]
        actions = (
            [Action(None, T.SHIFT, 0.0), Action(None, T.SWAP, 1.0)]
            + generate_actions(T.LEFT_ARC, self.relations.relations, ["nsubj"])
            + generate_actions(T.RIGHT_ARC, self.relations.relations, [])
        )

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {T.LEFT_ARC: 0, T.RIGHT_ARC: 1, T.SHIFT: 3, T.SWAP: 1}
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = configuration.select_actions(
            actions, costs, error_probability=0.0
        )

        assert best_action.transition == T.LEFT_ARC
        assert best_action.relation == "nsubj"
        assert best_wrong_action.transition == T.SWAP

        # Update oracle
        configuration.update_dynamic_oracle(best_action, shift_case)
        top_stack = configuration.top_stack_token
        assert top_stack.dependents == []
        assert top_stack.id not in top_stack.parent.dependents

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [4, 5, 6, 7, 8, 9, 0]
        assert configuration.stack == [1, 2]
        assert configuration.predicted_sentence[3].head == 4
        assert configuration.predicted_sentence[3].relation == "nsubj"

        # ===============================================================================
        # Iteration 5: LEFT ARC
        # ===============================================================================

        assert configuration.buffer == [4, 5, 6, 7, 8, 9, 0]
        assert configuration.stack == [1, 2]
        actions = (
            [Action(None, T.SHIFT, 0.0), Action(None, T.SWAP, 0.0)]
            + generate_actions(T.LEFT_ARC, self.relations.relations, ["aux", "rroot"])
            + generate_actions(T.RIGHT_ARC, self.relations.relations, [])
        )

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {T.LEFT_ARC: 0, T.RIGHT_ARC: 1, T.SHIFT: 2, T.SWAP: 1}
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = configuration.select_actions(
            actions, costs, error_probability=0.0
        )

        assert best_action.transition == T.LEFT_ARC
        assert best_action.relation == "aux"
        assert best_wrong_action.transition == T.LEFT_ARC
        assert best_wrong_action.relation == "rroot"

        # Update oracle
        configuration.update_dynamic_oracle(best_action, shift_case)
        top_stack = configuration.top_stack_token
        assert top_stack.dependents == []
        assert top_stack.id not in top_stack.parent.dependents

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [4, 5, 6, 7, 8, 9, 0]
        assert configuration.stack == [1]
        assert configuration.predicted_sentence[2].head == 4
        assert configuration.predicted_sentence[2].relation == "aux"

        # ===============================================================================
        # Iteration 6: LEFT ARC
        # ===============================================================================

        assert configuration.buffer == [4, 5, 6, 7, 8, 9, 0]
        assert configuration.stack == [1]
        actions = [
            Action(None, T.SHIFT, 0.0),
            Action(None, T.SWAP, 1.0),
        ] + generate_actions(T.LEFT_ARC, self.relations.relations, ["advmod"])

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {T.LEFT_ARC: 0, T.RIGHT_ARC: 1, T.SHIFT: 1, T.SWAP: 1}
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = configuration.select_actions(
            actions, costs, error_probability=0.0
        )

        assert best_action.transition == T.LEFT_ARC
        assert best_action.relation == "advmod"
        assert best_wrong_action.transition == T.SWAP

        # Update oracle
        configuration.update_dynamic_oracle(best_action, shift_case)
        top_stack = configuration.top_stack_token
        assert top_stack.dependents == []
        assert top_stack.id not in top_stack.parent.dependents

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [4, 5, 6, 7, 8, 9, 0]
        assert configuration.stack == []
        assert configuration.predicted_sentence[1].head == 4
        assert configuration.predicted_sentence[1].relation == "advmod"

        # ===============================================================================
        # Iteration 7: SHIFT
        # ===============================================================================

        assert configuration.buffer == [4, 5, 6, 7, 8, 9, 0]
        assert configuration.stack == []
        actions = [Action(None, T.SHIFT, 1.0)]

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {T.LEFT_ARC: 1, T.RIGHT_ARC: 1, T.SHIFT: 0, T.SWAP: 1}
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = configuration.select_actions(
            actions, costs, error_probability=0.0
        )

        assert best_action.transition == T.SHIFT
        assert best_wrong_action == Action.get_negative_action()

        # Update oracle
        sentence_before = deepcopy(configuration.sentence)
        configuration.update_dynamic_oracle(best_action, shift_case)
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [5, 6, 7, 8, 9, 0]
        assert configuration.stack == [4]
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # ===============================================================================
        # Iteration 8: SHIFT
        # ===============================================================================

        assert configuration.buffer == [5, 6, 7, 8, 9, 0]
        assert configuration.stack == [4]
        actions = [
            Action(None, T.SHIFT, 1.0),
            Action(None, T.SWAP, 1.0),
        ] + generate_actions(T.LEFT_ARC, self.relations.relations, [])

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {T.LEFT_ARC: 4, T.RIGHT_ARC: 1, T.SHIFT: 0, T.SWAP: 1}
        assert shift_case == 1

        # Get best actions
        best_action, best_valid_action, best_wrong_action = configuration.select_actions(
            actions, costs, error_probability=0.0
        )

        assert best_action.transition == T.SHIFT
        assert best_wrong_action.transition == T.SWAP

        # Update oracle
        sentence_before = deepcopy(configuration.sentence)
        configuration.update_dynamic_oracle(best_action, shift_case)
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [6, 7, 8, 9, 0]
        assert configuration.stack == [4, 5]
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # ===============================================================================
        # Iteration 9: SWAP
        # ===============================================================================

        assert configuration.buffer == [6, 7, 8, 9, 0]
        assert configuration.stack == [4, 5]
        actions = (
            [Action(None, T.SHIFT, 0.0), Action(None, T.SWAP, 1.0)]
            + generate_actions(T.LEFT_ARC, self.relations.relations, ["rroot"])
            + generate_actions(T.RIGHT_ARC, self.relations.relations, [])
        )

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {T.LEFT_ARC: 1, T.RIGHT_ARC: 1, T.SHIFT: 1, T.SWAP: 0}
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = configuration.select_actions(
            actions, costs, error_probability=0.0
        )

        assert best_action.transition == T.SWAP
        assert best_wrong_action.transition == T.LEFT_ARC
        assert best_wrong_action.relation == "rroot"

        # Update oracle
        sentence_before = deepcopy(configuration.sentence)
        configuration.update_dynamic_oracle(best_action, shift_case)
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [6, 5, 7, 8, 9, 0]
        assert configuration.stack == [4]
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # ===============================================================================
        # Iteration 10: SHIFT
        # ===============================================================================

        assert configuration.buffer == [6, 5, 7, 8, 9, 0]
        assert configuration.stack == [4]
        actions = [
            Action(None, T.SHIFT, 1.0),
            Action(None, T.SWAP, 1.0),
        ] + generate_actions(T.LEFT_ARC, self.relations.relations, [])

        # Compute costs
        costs, shift_case = configuration.get_transition_costs(actions)
        assert costs == {T.LEFT_ARC: 4, T.RIGHT_ARC: 1, T.SHIFT: 0, T.SWAP: 1}
        assert shift_case == 2

        # Get best actions
        best_action, best_valid_action, best_wrong_action = configuration.select_actions(
            actions, costs, error_probability=0.0
        )

        assert best_action.transition == T.SHIFT
        assert best_wrong_action.transition == T.SWAP

        # Update oracle
        sentence_before = deepcopy(configuration.sentence)
        configuration.update_dynamic_oracle(best_action, shift_case)
        assert sentences_are_equal(sentence_before, configuration.sentence)

        # Apply transition
        configuration.apply_transition(best_action)

        assert configuration.buffer == [5, 7, 8, 9, 0]
        assert configuration.stack == [4, 6]
        assert sentences_are_equal(sentence_before, configuration.sentence)
