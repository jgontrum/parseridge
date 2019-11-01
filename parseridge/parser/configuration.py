from random import random

import torch

from parseridge.corpus.sentence import Sentence
from parseridge.utils.helpers import Action, T, Transition
from parseridge.utils.logger import LoggerMixin


class Configuration(LoggerMixin):
    def __init__(
        self,
        sentence: Sentence,
        contextualized_input,
        model,
        predict=True,
        device=None,
        sentence_features=None,
    ):
        """
        The Configuration class is used to store the information about a
        parser configuration like stack, buffer, predictions etc.
        It also wraps many related methods like updating the oracle,
        computing transition costs and applying updates.
        Note that this *not* and immutable object and will therefore stay
        the same throughout the parsing of the corresponding sentence.

        Parameters
        ----------
        sentence : `Sentence` object
            The sentence to be parsed. In prediction mode, a dummy sentence
            should be passed here.
        contextualized_input : tensor
            The output of the LSTM for the sentence.
        model : torch.model
            The PyTorch model that is parsing this sentence.
        """
        self.model = model
        self.device = device if device else model.device if model else "cpu"
        self.sentence = sentence
        self.predicted_sentence = sentence.get_empty_copy() if predict else None
        self.contextualized_input = contextualized_input
        self.sentence_features = sentence_features
        self.scores = {}
        self.stack = []
        self.buffer = [token.id for token in sentence][1:] + [
            sentence[0].id
        ]  # Move the root token to the end
        self.num_swap = 0  # Used in prediction

        self.actions_history = []
        self.actions_hidden_state = None
        self.actions_cell_state = None
        self.finished_tokens = []

        self.decoder_hidden_state = None
        self.decoder_cell_state = None

    @property
    def left_arc_conditions(self):
        left_arc_conditions = len(self.stack) > 0

        if not self.in_training_mode:
            # In evaluation mode, make sure to avoid the multiple
            # root problem: Disallow left-arc from root
            # if stack has more than one element
            left_arc_conditions = left_arc_conditions and not (
                self.top_buffer_token.is_root and len(self.stack) > 1
            )

        return left_arc_conditions

    @property
    def right_arc_conditions(self):
        return len(self.stack) > 1

    @property
    def shift_conditions(self):
        return not self.top_buffer_token.is_root

    @property
    def swap_conditions(self):
        return len(self.stack) > 0 and self.top_stack_token.id < self.top_buffer_token.id

    def predict_actions(self):
        """
        For the current stack, buffer and the output of the MLP in
        `self.scores`, compute a list of possible actions together with their
        corresponding score.

        Returns
        -------
        actions : list of Action objects
            List of possible actions: A transition with its label and score.
        """
        assert self.scores, "Assign the output of the MLP first."

        # Now add all the possible transitions, together with
        # their score and the corresponding relation label
        # if it exists.
        actions = []
        if self.left_arc_conditions or self.right_arc_conditions:
            # To optimize the speed, we only add the gold transition if we
            # are in training mode and the transitions with the second best
            # label, since all other actions will be ignored anyhow.
            gold_relation_index = -1

            if self.in_training_mode:
                # Add actions using the correct relation for the current step
                gold_relation = self.top_stack_token.relation
                gold_relation_index = self.model.relations.signature.get_id(gold_relation)

                if self.left_arc_conditions:
                    actions.append(
                        Action(
                            relation=gold_relation,
                            transition=T.LEFT_ARC,
                            score=self.scores[T.LEFT_ARC][gold_relation_index],
                        )
                    )

                if self.right_arc_conditions:
                    actions.append(
                        Action(
                            relation=gold_relation,
                            transition=T.RIGHT_ARC,
                            score=self.scores[T.RIGHT_ARC][gold_relation_index],
                        )
                    )

            if self.left_arc_conditions:
                # Add the best left arc label action
                best_indices = self.scores[(T.LEFT_ARC, "best_scores_indices")]

                # In case we are in training mode and the gold index is
                # already the best, we chose the second best to generate
                # a 'wrong' example later on.
                ignore_best = best_indices[0] == gold_relation_index

                best_scores = self.scores[(T.LEFT_ARC, "best_scores")]
                best_score = best_scores[int(ignore_best)]
                best_index = best_indices[int(ignore_best)]

                best_label = self.model.relations.signature.get_item(best_index)

                actions.append(
                    Action(relation=best_label, transition=T.LEFT_ARC, score=best_score)
                )

            if self.right_arc_conditions:
                # Add the best right arc label action
                best_indices = self.scores[(T.RIGHT_ARC, "best_scores_indices")]

                # In case we are in training mode and the gold index is
                # already the best, we chose the second best to generate
                # a 'wrong' example later on.
                ignore_best = best_indices[0] == gold_relation_index

                best_scores = self.scores[(T.RIGHT_ARC, "best_scores")]
                best_score = best_scores[int(ignore_best)]
                best_index = best_indices[int(ignore_best)]

                best_label = self.model.relations.signature.get_item(best_index)

                actions.append(
                    Action(relation=best_label, transition=T.RIGHT_ARC, score=best_score)
                )

        if self.shift_conditions:
            tensor = self.scores[T.SHIFT][0]

            actions.append(Action(relation=None, transition=T.SHIFT, score=tensor))

        if self.swap_conditions:
            tensor = self.scores[T.SWAP][0]

            actions.append(Action(relation=None, transition=T.SWAP, score=tensor))

        return actions

    def get_transition_costs(self, actions_list):
        """
        Costs are used to exclude certain transitions are should not be
        performed at the current time. It excludes those that were not
        proposed by the `predict_actions()` method.

        Parameters
        ----------
        actions_list : list of Action objects
            Used to exclude those transitions that were not predicted in
            the first place.

        Returns
        -------
        costs : dict of Transition to int
            Maps a transition to its cost.
        shift_case : int
            Used in the dynamic oracle, because the SHIFT transition is
            possible in different cases.
        """

        # Group actions by transition
        actions = {
            transition: [
                action for action in actions_list if action.transition == transition
            ]
            for transition in Transition
        }

        # Set the cost to 1 by default
        costs = {transition: 1 for transition in Transition}
        shift_case = 0

        if actions[T.LEFT_ARC] and self.stack and self.buffer:
            cost = len(self.top_stack_token.dependents)
            cost += int(
                self.top_stack_token.parent.id != self.top_buffer_token.id
                and self.top_stack_token.id in self.top_stack_token.parent.dependents
            )

            costs[T.LEFT_ARC] = cost

        if actions[T.RIGHT_ARC] and len(self.stack) >= 2:
            second_stack_token = self.sentence[self.stack[-2]]

            cost = len(self.top_stack_token.dependents)
            cost += int(
                self.top_stack_token.parent != second_stack_token
                and self.top_stack_token.id in self.top_stack_token.parent.dependents
            )

            costs[T.RIGHT_ARC] = cost

        if actions[T.SHIFT] and self.buffer:
            rest_buffer_tokens = self.sentence[self.buffer[1:]]

            not_in_projected_order = [
                token
                for token in rest_buffer_tokens
                if token.projective_order < self.top_buffer_token.projective_order
                and token.id > self.top_buffer_token.id
            ]

            if not_in_projected_order:
                costs[T.SHIFT] = 0
                shift_case = 1

            else:
                stack_not_empty = len(self.stack) > 0

                buffer_dependents_in_stack = [
                    token_id
                    for token_id in self.top_buffer_token.dependents
                    if token_id in self.stack
                ]

                buffer_parent_in_stack = (
                    self.top_buffer_token.parent.id in self.stack[:-1]
                    and self.top_buffer_token.id in self.top_buffer_token.parent.dependents
                )

                costs[T.SHIFT] = len(buffer_dependents_in_stack) + int(
                    stack_not_empty and buffer_parent_in_stack
                )

                shift_case = 2

        if actions[T.SWAP] and self.stack and self.buffer:
            first_stack_token = self.sentence[self.stack[-1]]
            first_buffer_token = self.sentence[self.buffer[0]]

            if first_stack_token.projective_order > first_buffer_token.projective_order:
                # SWAP has priority, so disable all other options
                costs = {k: 1 for k in costs.keys()}
                costs[T.SWAP] = 0

        return costs, shift_case

    def get_valid_actions(self, actions, costs):
        valid_actions = []
        for action in actions:
            # Only take transitions with a cost of 0 into account
            if costs[action.transition] > 0:
                continue

            if action.transition in [T.SHIFT, T.SWAP]:
                # If the transition is SHIFT or SWAP,
                # we don't have to check a relation and can
                # accept them right away
                valid_actions.append(action)

            elif self.stack and action.relation == self.top_stack_token.relation:
                # Otherwise, the relation must match the gold
                # relation for the first item on the stack.
                valid_actions.append(action)

        return valid_actions

    def get_wrong_actions(self, actions, costs):
        wrong_actions = []
        for action in actions:
            if costs[action.transition] != 0 or (
                action.transition != T.SHIFT
                and action.transition != T.SWAP
                and action.relation != self.top_stack_token.relation
            ):
                wrong_actions.append(action)

        return wrong_actions

    def select_actions(self, actions, costs, error_probability=0.0, margin_threshold=2.5):
        """
        Given the predicted actions and the costs for the transitions,
        find the best action and the best wrong action. Both are needed
        to calculate the margin loss. To make the parser more robust,
        we sometimes chose to follow the 'wrong' transition.

        Parameters
        ----------
        actions : list of Action objects
            Proposed actions - Transitions with labels and scores
        costs : dict of Transition to int
            Maps a transition to its cost.

        Returns
        -------
        best_action : Action object
            The action to follow. Is either `best_valid_action` or
            `best_wrong_action`.
        best_valid_action : Action object
            Action with the highest score that has costs of 0 and matches
            the correct relation label.
        best_wrong_action : Action object
            Action with the highest score that is not a valid action.
        """
        # Given the classification scores and the costs, split
        # the possible actions into valid and invalid ones.
        # Invalid actions might be technically allowed in the
        # current situation, but do not make sense given the
        # gold dependency tree and would lead to errors.
        # Get the best valid transition

        valid_actions = self.get_valid_actions(actions, costs)
        best_valid_action = self.get_best_action(valid_actions)

        wrong_actions = self.get_wrong_actions(actions, costs)
        if wrong_actions:
            best_wrong_action = self.get_best_action(wrong_actions)
        else:
            # Add one negative action to make sure we have at least one
            # 'wrong' action to calculate the error against.
            best_wrong_action = Action.get_negative_action()

        # To make sure the model keeps learning and to make it more
        # robust, we let it make wrong decisions from time to time.
        # The better the model becomes, the higher is the chance
        # of introducing a wrong decision. However, if there is
        # any chance of making a SWAP transition, skip this step.

        best_action = best_valid_action
        no_swap_possible = costs[T.SWAP] != 0 and best_wrong_action.transition != T.SWAP

        is_valid_transition = best_wrong_action.transition is not None

        if (
            no_swap_possible
            and is_valid_transition
            and random() <= error_probability
            and best_action.np_score > best_wrong_action.np_score + margin_threshold
        ):
            best_action = best_wrong_action

        return best_action, best_valid_action, best_wrong_action

    def update_dynamic_oracle(self, action, shift_case):
        """
        Given the chosen action, update the dependency graph accordingly.

        Parameters
        ----------
        action : Action object
            The chosen action. Can be the best valid or best invalid.
        shift_case : int
            Output of the `get_transition_costs()` method.
        """
        assert 0 <= shift_case <= 2
        assert action.transition is not None

        if action.transition == T.SHIFT and shift_case == 2 and self.buffer:
            # Remove all references to tokens in the stack
            first_buffer_token = self.sentence[self.buffer[0]]
            first_buffer_token_parent = first_buffer_token.parent

            parent_in_stack = first_buffer_token_parent.id in self.stack[:-1]
            token_is_parent_dep = (
                first_buffer_token.id in first_buffer_token_parent.dependents
            )

            # Remove the parent from the first buffer token if in the stack
            if parent_in_stack and token_is_parent_dep:
                first_buffer_token_parent.dependents.remove(first_buffer_token.id)

            # Remove dependents of the first buffer token that are in the stack
            first_buffer_token.dependents = [
                dep_id
                for dep_id in first_buffer_token.dependents
                if dep_id not in self.stack
            ]

        elif action.transition in [T.LEFT_ARC, T.RIGHT_ARC] and self.stack:
            first_buffer_token = self.sentence[self.stack[-1]]
            first_buffer_token.dependents = []

            # Remove the token from its parent's dependents list
            if first_buffer_token.id in first_buffer_token.parent.dependents:
                first_buffer_token.parent.dependents.remove(first_buffer_token.id)

    def apply_transition(self, action):
        """
        Updates the data structures by applying the transition in the given
        action. This only affects the `predicted_sentence`.

        Parameters
        ----------
        action : Action object
            Contains the transition and label if required.
        """
        assert action.transition is not None

        dependent = parent = None
        if action.transition == T.SHIFT:
            self.stack.append(self.buffer[0])
            del self.buffer[0]

        elif action.transition == T.SWAP:
            dependent = self.stack.pop()
            self.buffer.insert(1, dependent)

        elif action.transition == T.LEFT_ARC:
            dependent = self.stack.pop()
            parent = self.buffer[0]

        elif action.transition == T.RIGHT_ARC:
            dependent = self.stack.pop()
            parent = self.stack[-1]

        if action.transition in [T.LEFT_ARC, T.RIGHT_ARC] and self.predicted_sentence:
            # Make an attachment in the tree
            self.predicted_sentence[dependent].head = parent
            self.predicted_sentence[dependent].relation = action.relation
            self.predicted_sentence[parent].dependents.append(dependent)

            self.finished_tokens.append(dependent)

        self.actions_history.append(action)

    @staticmethod
    def get_best_action(actions):
        """
        Given a list of `Action` objects, select the best one.
        This is a helper, because the `max()` function is a bottleneck
        in the parser and hopefully we will find a more efficient solution.

        Parameters
        ----------
        actions : list of Action object
            The actions to chose the best one from.
        Returns
        -------
        Action object
        """
        if len(actions) == 1:
            return actions[0]

        return max(actions, key=lambda action: action.np_score)

    @property
    def is_terminal(self):
        return not self.stack and len(self.buffer) == 1

    @property
    def swap_possible(self):
        return self.num_swap < 2 * len(self.sentence)

    @property
    def top_buffer_token(self):
        return self.sentence[self.buffer[0]]

    @property
    def top_stack_token(self):
        return self.sentence[self.stack[-1]]

    @property
    def in_training_mode(self):
        return False if self.model is None else self.model.training

    @property
    def finished_tokens_tensor(self):
        return torch.tensor(self.finished_tokens, dtype=torch.int64, device=self.device)

    @property
    def reversed_finished_tokens_tensor(self):
        return torch.tensor(
            self.finished_tokens[::-1], dtype=torch.int64, device=self.device
        )

    @property
    def stack_tensor(self):
        return torch.tensor(self.stack, dtype=torch.int64, device=self.device)

    @property
    def buffer_tensor(self):
        return torch.tensor(self.buffer, dtype=torch.int64, device=self.device)

    @property
    def actions_tensor(self):
        actions = [action.transition.value for action in self.actions_history]
        return torch.tensor(actions, dtype=torch.int64, device=self.device)
