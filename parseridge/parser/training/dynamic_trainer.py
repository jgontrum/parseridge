from typing import List, Optional, Tuple, Union

import torch
from torch.nn.modules.loss import _Loss

from parseridge.corpus.corpus import CorpusIterator, Corpus
from parseridge.corpus.sentence import Sentence
from parseridge.corpus.training_data import ConLLDataset
from parseridge.parser.configuration import Configuration
from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.utils import (
    pad_list_of_lists,
    to_int_tensor,
    pad_tensor_list,
)
from parseridge.parser.training.base_trainer import Trainer
from parseridge.parser.training.callbacks.base_callback import StopEpoch, StopTraining
from parseridge.parser.training.hyperparameters import Hyperparameters
from parseridge.utils.helpers import T


class DynamicTrainer(Trainer):
    """
    The default trainer.
    """

    def fit(
        self,
        epochs: int,
        training_data: Union[Corpus, ConLLDataset],
        hyper_parameters: Hyperparameters = None,
        batch_size: int = 4,
        oov_probability: float = 0.00,
        token_dropout: float = 0.00,
        margin_threshold: float = 1.00,
        error_probability: float = 0.1,
        update_frequency: int = 50,
        criterion: Optional[_Loss] = None,
        **kwargs,
    ) -> None:
        if not isinstance(training_data, Corpus):
            raise ValueError(f"The DynamicTrainer requires a Corpus object for training.")
        if hyper_parameters:
            raise ValueError(f"Hyper Parameter objects are not supported here.")

        initial_epoch = self.last_epoch

        self.callback_handler.on_train_begin(
            epochs=epochs + initial_epoch, batch_size=batch_size
        )

        for epoch in range(initial_epoch + 1, epochs + initial_epoch + 1):
            try:
                self._run_epoch(
                    epoch=epoch,
                    training_data=training_data,
                    criterion=criterion,
                    batch_size=batch_size,
                    oov_probability=oov_probability,
                    token_dropout=token_dropout,
                    margin_threshold=margin_threshold,
                    error_probability=error_probability,
                    update_frequency=update_frequency,
                )
            except StopTraining:
                self.logger.info(f"Stopping training after {epoch} epochs.")
                break

        self.callback_handler.on_train_end()

    def _run_epoch(
        self,
        epoch: int,
        training_data: Corpus,
        batch_size: int,
        criterion: Optional[_Loss],
        update_frequency: int,
        oov_probability: float,
        token_dropout: float,
        margin_threshold: float,
        error_probability: float,
    ):
        iterator = CorpusIterator(
            training_data,
            batch_size=batch_size,
            shuffle=True,
            train=True,
            oov_probability=oov_probability,
            group_by_length=True,
            token_dropout=token_dropout,
        )

        self.callback_handler.on_epoch_begin(
            epoch=epoch, num_batches=len(iterator), training_data=training_data
        )

        loss: List[torch.Tensor] = []
        epoch_loss = 0

        for i, batch in enumerate(iterator):
            try:
                self.callback_handler.on_batch_begin(batch=i, batch_data=batch)

                current_loss = self._process_training_batch(
                    batch=batch,
                    error_probability=error_probability,
                    margin_threshold=margin_threshold,
                    criterion=criterion,
                )

                if not criterion:
                    loss += current_loss

                    if len(loss) > update_frequency:
                        combined_loss = sum(loss) / len(loss)
                        self.learn(combined_loss)

                        batch_loss = combined_loss.item()
                        batch_loss += margin_threshold * len(loss)

                        epoch_loss += batch_loss
                        loss = []
                    else:
                        batch_loss = None
                else:
                    loss.append(current_loss)
                    if len(loss) > 10:
                        combined_loss = sum(loss) / len(loss)
                        self.learn(combined_loss)
                        batch_loss = combined_loss.item()
                        epoch_loss += batch_loss
                        loss = []
                    else:
                        batch_loss = None
                self.last_epoch = epoch

                self.callback_handler.on_batch_end(
                    batch=i, batch_data=batch, batch_loss=batch_loss
                )
            except StopEpoch:
                self.logger.info(f"Stopping epoch after {i}/{len(iterator)} batches.")
                break

        self.callback_handler.on_epoch_end(epoch=epoch, epoch_loss=epoch_loss)

    def _process_training_batch(
        self,
        batch: Tuple[torch.Tensor, List[Sentence]],
        error_probability: float,
        margin_threshold: float,
        criterion: Optional[_Loss] = None,
    ) -> List[torch.Tensor]:
        """
        Parses the sentences in the given batch and returns the loss values.
        """
        loss = []

        transition_logits = []
        transition_gold_labels = []

        relation_logits = []
        relation_gold_labels = []

        # The batch contains of a tensor of processed sentences
        # that are ready to be used as an input to the LSTM
        # and the corresponding sentence objects that are needed
        # to grade the performance of the predictions.
        sentence_features, sentences = batch

        # Run the sentence through the encoder to get the outputs.
        # These outputs will stay the same for the sentence,
        # so we compute them once in the beginning.
        token_sequences = sentence_features[:, 0, :]

        sentence_lengths = to_int_tensor(
            data=[len(sentence) for sentence in sentences], device=self.model.device
        )

        contextualized_tokens_batch = self.model.get_contextualized_input(
            token_sequences, sentence_lengths
        )

        # Create the initial configurations for all sentences in the batch
        configurations = [
            Configuration(
                sentence,
                contextualized_input,
                self.model,
                sentence_features=sentence_feature,
            )
            for contextualized_input, sentence_feature, sentence in zip(
                contextualized_tokens_batch, sentence_features, sentences
            )
        ]

        # Main loop for the sentences in this batch
        while configurations:
            # Remove all finished configurations
            configurations = [c for c in configurations if not c.is_terminal]
            if not configurations:
                break

            # Pass the stacks and buffers through the MLPs in one batch
            configurations = self.predict_logits(configurations, self.model)

            # The actual computation of the loss must be done sequentially
            for configuration in configurations:
                # Predict a list of possible actions: Transitions, their
                # label (if the transition is LEFT/ RIGHT_ARC) and the
                # score of the action based on the MLP output.
                actions = configuration.predict_actions()

                # Calculate the 'costs' for each action. These determine
                # which action should be performed based on the given
                # conf
                costs, shift_case = configuration.get_transition_costs(actions)

                # Compute the best valid and the best wrong action,
                # where the latter on is a transition that is technically
                # possible, but would introduce an error compared to the
                # gold tree. To keep the model robust, we sometimes
                # decided, however, to use it instead of the valid one.
                best_action, best_valid_action, best_wrong_action = configuration.select_actions(
                    actions, costs, error_probability, margin_threshold
                )

                # Apply the dynamic oracle to update the sentence structure
                # for the case that the chosen action does not exactly
                # follow the gold tree.
                configuration.update_dynamic_oracle(best_action, shift_case)

                # Apply the best action and update the stack and buffer
                configuration.apply_transition(best_action)

                if criterion:
                    # Compute CrossEntropy loss
                    gold_transition = best_valid_action.transition.value
                    gold_relation = self.model.relations.label_signature.get_id(
                        (best_valid_action.transition, best_valid_action.relation)
                    )

                    transition_logits.append(configuration.scores["transition_logits"])
                    relation_logits.append(configuration.scores["relation_logits"])

                    transition_gold_labels.append(gold_transition)
                    relation_gold_labels.append(gold_relation)
                else:
                    # Compute the loss by using the margin between the scores
                    if (
                        best_wrong_action.transition is not None
                        and best_valid_action.np_score
                        < best_wrong_action.np_score + margin_threshold
                    ):
                        margin = best_wrong_action.score - best_valid_action.score
                        loss.append(margin)

        if criterion:
            transition_logits = torch.stack(transition_logits)
            relation_logits = torch.stack(relation_logits)

            transition_gold_labels = to_int_tensor(
                transition_gold_labels, self.model.device
            )
            relation_gold_labels = to_int_tensor(relation_gold_labels, self.model.device)

            transition_loss = criterion(transition_logits, transition_gold_labels)
            relation_loss = criterion(relation_logits, relation_gold_labels)

            return transition_loss + relation_loss
        return loss

    @staticmethod
    def predict_logits(configurations: List[Configuration], model: Module):
        """
        Wraps the stacks, buffers and contextualized token data of all
        given configurations into a tensor which is then passed through
        the MLP to compute the classification scores in the configurations.
        """

        contextualized_inputs = [c.contextualized_input for c in configurations]

        padded_stacks = pad_list_of_lists([list(reversed(c.stack)) for c in configurations])
        padded_buffers = pad_list_of_lists([c.buffer for c in configurations])
        padded_finished_tokens = pad_tensor_list(
            [c.reversed_finished_tokens_tensor for c in configurations]
        )
        padded_sentence_features = (
            torch.stack([c.sentence_features for c in configurations])
            if configurations[0].sentence_features is not None
            else None
        )
        sentence_ids = [c.sentence.id for c in configurations]

        stack_len = [len(c.stack) for c in configurations]
        buffer_len = [len(c.buffer) for c in configurations]
        finished_tokens_len = [len(c.finished_tokens) for c in configurations]
        sentence_len = [len(c.sentence) for c in configurations]

        transition_logits, relation_logits = model.compute_mlp_output(
            contextualized_input_batch=torch.stack(contextualized_inputs),
            stacks=to_int_tensor(padded_stacks, device=model.device),
            stack_lengths=to_int_tensor(stack_len, device=model.device),
            buffers=to_int_tensor(padded_buffers, device=model.device),
            buffer_lengths=to_int_tensor(buffer_len, device=model.device),
            finished_tokens=to_int_tensor(padded_finished_tokens, device=model.device),
            finished_tokens_lengths=to_int_tensor(finished_tokens_len, device=model.device),
            sentence_lengths=to_int_tensor(sentence_len, device=model.device),
            sentence_features=padded_sentence_features,
            sentence_ids=sentence_ids,
        )

        # Isolate the columns for the transitions
        left_arc = transition_logits[:, T.LEFT_ARC.value].view(-1, 1)
        right_arc = transition_logits[:, T.RIGHT_ARC.value].view(-1, 1)
        shift = transition_logits[:, T.SHIFT.value].view(-1, 1)
        swap = transition_logits[:, T.SWAP.value].view(-1, 1)

        # Isolate the columns for the different labels
        relation_slices = model.relations.slices
        shift_labels = relation_logits[:, relation_slices[T.SHIFT]]
        swap_labels = relation_logits[:, relation_slices[T.SWAP]]
        left_arc_labels = relation_logits[:, relation_slices[T.LEFT_ARC]]
        right_arc_labels = relation_logits[:, relation_slices[T.RIGHT_ARC]]

        # Add them in one batch
        shift_score_batch = torch.add(shift, shift_labels)
        swap_score_batch = torch.add(swap, swap_labels)
        left_arc_scores_batch = torch.add(left_arc, left_arc_labels)
        right_arc_scores_batch = torch.add(right_arc, right_arc_labels)

        # For the left and right arc scores, we're only interested in the
        # two best entries, so we extract then in one go.
        left_arc_scores_sorted, left_arc_scores_indices = torch.sort(
            left_arc_scores_batch, descending=True
        )
        right_arc_scores_sorted, right_arc_scores_indices = torch.sort(
            right_arc_scores_batch, descending=True
        )

        # Only take the best two items
        left_arc_scores_sorted = left_arc_scores_sorted[:, :2]
        right_arc_scores_sorted = right_arc_scores_sorted[:, :2]

        # We need them later in RAM, so retrieve them all at once from the gpu
        left_arc_scores_indices = left_arc_scores_indices[:, :2].cpu().numpy()
        right_arc_scores_indices = right_arc_scores_indices[:, :2].cpu().numpy()

        combinations = zip(
            configurations,
            shift_score_batch,
            swap_score_batch,
            left_arc_scores_batch,
            right_arc_scores_batch,
            left_arc_scores_sorted,
            left_arc_scores_indices,
            right_arc_scores_sorted,
            right_arc_scores_indices,
            transition_logits,
            relation_logits,
        )

        # Update the result of the classifiers in the configurations
        for combination in combinations:
            configuration = combination[0]

            configuration.scores = {
                T.SHIFT: combination[1],
                T.SWAP: combination[2],
                T.LEFT_ARC: combination[3],
                T.RIGHT_ARC: combination[4],
                (T.LEFT_ARC, "best_scores"): combination[5],
                (T.LEFT_ARC, "best_scores_indices"): combination[6],
                (T.RIGHT_ARC, "best_scores"): combination[7],
                (T.RIGHT_ARC, "best_scores_indices"): combination[8],
                "transition_logits": combination[9],
                "relation_logits": combination[10],
            }

        return configurations
