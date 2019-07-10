from typing import List

import torch

from parseridge.corpus.corpus import CorpusIterator, Corpus
from parseridge.parser.configuration import Configuration
from parseridge.parser.modules.data_parallel import Module
from parseridge.parser.modules.utils import pad_list_of_lists, to_int_tensor
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
        training_data: Corpus,
        hyper_parameters: Hyperparameters = None,
        **kwargs,
    ) -> None:
        if not isinstance(training_data, Corpus):
            raise ValueError(f"The DynamicTrainer requires a Corpus object for training.")

        hyper_parameters = (hyper_parameters or Hyperparameters()).update(**kwargs)

        initial_epoch = self.last_epoch

        self.callback_handler.on_train_begin(
            epochs=epochs + initial_epoch, hyper_parameters=hyper_parameters
        )

        for epoch in range(initial_epoch + 1, epochs + initial_epoch + 1):
            try:
                self._run_epoch(epoch, training_data, hyper_parameters)
            except StopTraining:
                self.logger.info(f"Stopping training after {epoch} epochs.")
                break

        self.callback_handler.on_train_end()

    def _run_epoch(
        self, epoch: int, training_data: Corpus, hyper_parameters: Hyperparameters
    ):
        iterator = CorpusIterator(
            training_data,
            batch_size=hyper_parameters.batch_size,
            shuffle=True,
            train=True,
            oov_probability=hyper_parameters.oov_probability,
            group_by_length=True,
            token_dropout=hyper_parameters.token_dropout,
        )

        self.callback_handler.on_epoch_begin(
            epoch=epoch, num_batches=len(iterator), training_data=training_data
        )

        loss = []
        epoch_loss = 0

        for i, batch in enumerate(iterator):
            try:
                self.callback_handler.on_batch_begin(batch=i, batch_data=batch)

                current_loss = self._process_training_batch(
                    batch=batch,
                    error_probability=hyper_parameters.error_probability,
                    margin_threshold=hyper_parameters.margin_threshold,
                )

                loss += current_loss

                if len(loss) > 50:
                    combined_loss = sum(loss) / len(loss)
                    self.learn(combined_loss)

                    batch_loss = combined_loss.item() + (
                        hyper_parameters.margin_threshold * len(loss)
                    )
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
        self, batch, error_probability, margin_threshold
    ) -> List[torch.Tensor]:
        loss = []

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
            Configuration(sentence, contextualized_input, self.model)
            for contextualized_input, sentence in zip(
                contextualized_tokens_batch, sentences
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

                # Compute the loss by using the margin between the scores
                if (
                    best_wrong_action.transition is not None
                    and best_valid_action.np_score
                    < best_wrong_action.np_score + margin_threshold
                ):
                    margin = best_wrong_action.score - best_valid_action.score
                    loss.append(margin)

        return loss

    @staticmethod
    def predict_logits(configurations: List[Configuration], model: Module):
        """
        Wraps the stacks, buffers and contextualized token data of all
        given configurations into a tensor which is then passed through
        the MLP to compute the classification scores in the configurations.
        :param configurations: List of not finished Configurations
        :return: Updated Configurations
        """

        contextualized_inputs = [c.contextualized_input for c in configurations]

        padded_stacks = pad_list_of_lists([list(reversed(c.stack)) for c in configurations])
        padded_buffers = pad_list_of_lists([c.buffer for c in configurations])

        stack_len = [len(c.stack) for c in configurations]
        buffer_len = [len(c.buffer) for c in configurations]

        clf_transitions, clf_labels = model.compute_mlp_output(
            contextualized_input_batch=torch.stack(contextualized_inputs),
            stacks=to_int_tensor(padded_stacks, device=model.device),
            stack_lengths=to_int_tensor(stack_len, device=model.device),
            buffers=to_int_tensor(padded_buffers, device=model.device),
            buffer_lengths=to_int_tensor(buffer_len, device=model.device),
        )

        # Isolate the columns for the transitions
        left_arc = clf_transitions[:, T.LEFT_ARC.value].view(-1, 1)
        right_arc = clf_transitions[:, T.RIGHT_ARC.value].view(-1, 1)
        shift = clf_transitions[:, T.SHIFT.value].view(-1, 1)
        swap = clf_transitions[:, T.SWAP.value].view(-1, 1)

        # Isolate the columns for the different labels
        relation_slices = model.relations.slices
        shift_labels = clf_labels[:, relation_slices[T.SHIFT]]
        swap_labels = clf_labels[:, relation_slices[T.SWAP]]
        left_arc_labels = clf_labels[:, relation_slices[T.LEFT_ARC]]
        right_arc_labels = clf_labels[:, relation_slices[T.RIGHT_ARC]]

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
            }

        return configurations
