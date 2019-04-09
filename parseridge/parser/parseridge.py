from datetime import datetime
from time import time

import torch
from tqdm import tqdm

from parseridge.corpus.corpus import CorpusIterator
from parseridge.parser.configuration import Configuration
from parseridge.parser.model import ParseridgeModel
from parseridge.parser.trainer import Trainer
from parseridge.utils.evaluate import CoNNLEvaluator
from parseridge.utils.helpers import T, Metric
from parseridge.utils.logger import LoggerMixin


class ParseRidge(LoggerMixin):
    """
    The core engine of the parser. Here, the training and prediction parts
    are implemented.
    """

    def __init__(self, device):
        self.stats = []
        self.time_prefix = lambda: datetime.now().strftime("%Y%m%d-%H%M%S")

        self.model = None
        self.trainer = None
        self.id_ = str(self.time_prefix())
        self.device = device

    def fit(self, corpus, relations, dev_corpus=None, num_stack=3, num_buffer=1,
            embedding_size=100, lstm_hidden_size=125, lstm_layers=2,
            relation_mlp_layers=None, transition_mlp_layers=None, margin_threshold=2.5,
            error_probability=0.1, oov_probability=0.25, token_dropout=0.01,
            lstm_dropout=0.33, mlp_dropout=0.25, batch_size=4, num_epochs=3,
            gradient_clipping=10.0, weight_decay=0.0, learning_rate=0.001,
            update_size=50, loss_factor=0.75, loss_strategy="avg"):

        self.model = ParseridgeModel(
            relations=relations,
            vocabulary=corpus.vocabulary,
            num_stack=num_stack,
            num_buffer=num_buffer,
            lstm_dropout=lstm_dropout,
            mlp_dropout=mlp_dropout,
            embedding_size=embedding_size,
            lstm_hidden_size=lstm_hidden_size,
            lstm_layers=lstm_layers,
            transition_mlp_layers=transition_mlp_layers,
            relation_mlp_layers=relation_mlp_layers,
            device=self.device
        ).to(self.device)

        self.trainer = Trainer(
            self.model,
            learning_rate=learning_rate,
            gradient_clipping=gradient_clipping,
            weight_decay=weight_decay,
            mode=loss_strategy,
            loss_factor=loss_factor,
            update_size=update_size
        )

        for epoch in range(num_epochs):
            t0 = time()
            self.logger.info(f"Starting epoch #{epoch + 1}...")

            epoch_metric = self._run_epoch(
                corpus=corpus,
                batch_size=batch_size,
                error_probability=error_probability,
                margin_threshold=margin_threshold,
                oov_probability=oov_probability,
                token_dropout=token_dropout
            )

            self.logger.info(f"Epoch loss: {epoch_metric.loss / len(corpus):.8f}")
            self.logger.info(f"Updates: {epoch_metric.num_updates}")
            self.logger.info(f"Back propagations: {epoch_metric.num_backprop}")
            self.logger.info(f"Induced errors: {epoch_metric.num_errors}")

            # Evaluate on training corpus
            train_scores = CoNNLEvaluator().get_las_score_for_sentences(
                *self.predict(corpus))
            self.logger.info(
                f"Performance on the training set after {epoch + 1} epochs: "
                f"LAS: {train_scores['LAS']:.2f} | "
                f"UAS: {train_scores['UAS']:.2f}"
            )

            # Evaluate on dev corpus
            if dev_corpus:
                dev_scores = CoNNLEvaluator().get_las_score_for_sentences(
                    *self.predict(dev_corpus))

                self.logger.info(
                    f"Performance on the dev set after {epoch + 1} epochs: "
                    "     "
                    f"LAS: {dev_scores['LAS']:.2f} | "
                    f"UAS: {dev_scores['UAS']:.2f}"
                )

            duration = time() - t0
            self.logger.info(
                f"Finished epoch in "
                f"{int(duration / 60)}:{int(duration % 60):01} minutes.")

    def _run_epoch(self, corpus, batch_size=4, error_probability=0.1,
                   margin_threshold=2.5, oov_probability=0.25,
                   token_dropout=0.01, update_pbar_interval=50):
        """
        Wrapper that trains the model on the whole data set once.

        Parameters
        ----------
        corpus : Corpus object
            The training corpus.
        optimizer : torch.optim object
            The optimizer for the parser model `self.model`.

        Returns
        -------
        epoch_metric : Metric object
            Contains statistics about the performance of this epoch.
        """

        # Set our model into training mode which enables dropout and
        # the accumulation of gradients
        self.model.train()
        self.model.before_epoch()

        loss = []

        epoch_metric = Metric()
        interval_metric = Metric()

        iterator = CorpusIterator(
            corpus,
            batch_size=batch_size,
            shuffle=True,
            train=True,
            oov_probability=oov_probability,
            group_by_length=True,
            token_dropout=token_dropout
        )
        with tqdm(total=len(corpus)) as pbar:
            pbar_template = (
                "Batch Loss: {loss:8.4f} | Updates: {updates:5.1f} | "
                "Induced Errors: {errors:3.1f}"
            )
            pbar.set_description(pbar_template.format(
                loss=0, updates=0, errors=0
            ))
            for batch in iterator:
                loss, batch_metric = self._run_training_batch(
                    batch, loss, error_probability, margin_threshold)

                epoch_metric += batch_metric
                interval_metric += batch_metric

                num_sentences = interval_metric.iterations * batch_size
                if num_sentences >= update_pbar_interval and interval_metric.num_updates:
                    assert num_sentences > 0
                    # Update the progress bar less frequently

                    desc = pbar_template.format(
                        loss=interval_metric.loss / interval_metric.num_updates,
                        updates=interval_metric.num_updates / num_sentences,
                        errors=interval_metric.num_errors / num_sentences,
                    )
                    pbar.set_description(desc)
                    interval_metric = Metric()

                pbar.update(len(batch[1]))

        # In case there are some margin losses left, compute their gradients.
        _, epoch_metric = self.trainer.learn(loss, epoch_metric)
        self.model.after_epoch()
        return epoch_metric

    def _run_training_batch(self, batch, loss, error_probability, margin_threshold):
        """
        Trains the parser model on the data given in `batch` and performs
        back-propagation, if the number of updates is above a certain
        threshold.

        Parameters
        ----------
        batch : tuple of tensor and list of Sentence
            A slice of the training data to train on.
        loss : list of tensors
            Back-propagation is only performed when a certain number of updates
            has been made. If the preceding batch did not update the weights,
            the losses are passed to this batch.

        Returns
        -------
        loss : list of tensors
            If no back-propagation has been executed, it is a list of margin
            tensors that needs to be passed to the next batch.
            Otherwise empty.
        batch_metric : Metric object
            Object that contains statistics about the executed batch like
            the loss, number of errors and number of updates.
        """
        self.model.before_batch()

        # The batch contains of a tensor of processed sentences
        # that are ready to be used as an input to the LSTM
        # and the corresponding sentence objects that are needed
        # to grade the performance of the predictions.
        sentence_features, sentences = batch

        # Collect all kinds of information here
        batch_metric = Metric()

        # Run the sentence through the LSTM to get the outputs.
        # These outputs will stay the same for the sentence,
        # so we compute them once in the beginning.
        contextualized_tokens_batch = self.model.compute_lstm_output(
            sentences, sentence_features
        )

        # Create the initial configurations for all sentences in the batch
        configurations = [
            Configuration(sentence, contextualized_input, self.model)
            for contextualized_input, sentence in
            zip(contextualized_tokens_batch, sentences)
        ]

        # Main loop for the sentences in this batch
        while configurations:
            # Remove all finished configurations
            configurations = [c for c in configurations if not c.is_terminal]
            if not configurations:
                break

            # Pass the stacks and buffers through the MLPs in one batch
            configurations = self._update_classification_scores(configurations)
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
                best_action, best_valid_action, best_wrong_action = \
                    configuration.select_actions(
                        actions, costs, error_probability, margin_threshold)

                # Apply the dynamic oracle to update the sentence structure
                # for the case that the chosen action does not exactly
                # follow the gold tree.
                configuration.update_dynamic_oracle(best_action, shift_case)

                # Apply the best action and update the stack and buffer
                configuration.apply_transition(best_action)

                # Compute the loss by using the margin between the scores
                if (best_wrong_action.transition is not None
                        and best_valid_action.np_score <
                        best_wrong_action.np_score + margin_threshold):
                    margin = best_wrong_action.score - best_valid_action.score
                    batch_metric.num_updates += 1
                    loss.append(margin)

                if best_action.transition in [T.LEFT_ARC, T.RIGHT_ARC]:
                    batch_metric.num_transitions += 1
                if best_action != best_valid_action:
                    batch_metric.num_errors += 1

        # Perform back propagation
        loss, batch_metric = self.trainer.learn(loss, batch_metric)
        self.model.after_batch()
        return loss, batch_metric

    def predict(self, corpus, batch_size=512, remove_pbar=True):
        self.model = self.model.eval()

        gold_sentences = []
        pred_sentences = []

        iterator = CorpusIterator(corpus, batch_size=batch_size, train=False)
        with tqdm(
                total=len(corpus),
                desc=f"Predicting sentences...",
                leave=not remove_pbar
        ) as pbar:
            for batch in iterator:
                pred, gold = self._run_prediction_batch(batch)
                pred_sentences += pred
                gold_sentences += gold

                pbar.update(len(batch[1]))

        return gold_sentences, pred_sentences

    def _run_prediction_batch(self, batch):
        pred_sentences = []
        gold_sentences = []

        sentence_features, sentences = batch

        contextualized_tokens_batch = self.model.compute_lstm_output(
            sentences, sentence_features
        )

        configurations = [
            Configuration(sentence, contextualized_input, self.model)
            for contextualized_input, sentence in
            zip(contextualized_tokens_batch, sentences)
        ]

        while configurations:
            # Pass the stacks and buffers through the MLPs in one batch
            configurations = self._update_classification_scores(
                configurations)

            # The actual computation of the loss must be done sequentially
            for configuration in configurations:
                # Predict a list of possible actions: Transitions, their
                # label (if the transition is LEFT/ RIGHT_ARC) and the
                # score of the action based on the MLP output.
                actions = configuration.predict_actions()

                if not configuration.swap_possible:
                    # Exclude swap options
                    actions = [
                        action for action in actions
                        if action.transition != T.SWAP
                    ]

                assert actions
                best_action = Configuration.get_best_action(actions)

                if best_action.transition == T.SWAP:
                    configuration.num_swap += 1

                configuration.apply_transition(best_action)

                if configuration.is_terminal:
                    pred_sentences.append(configuration.predicted_sentence)
                    gold_sentences.append(configuration.sentence)

            # Remove all finished configurations
            configurations = [c for c in configurations if not c.is_terminal]

        return pred_sentences, gold_sentences

    def _update_classification_scores(self, configurations):
        """
        Wraps the stacks, buffers and contextualized token data of all
        given configurations into a tensor which is then passed through
        the MLP to compute the classification scores in the configurations.
        :param configurations: List of not finished Configurations
        :return: Updated Configurations
        """
        clf_transitions, clf_labels = self.model.compute_mlp_output(
            [c.contextualized_input for c in configurations],
            [c.stack for c in configurations],
            [c.buffer for c in configurations]
        )

        # Isolate the columns for the transitions
        left_arc = clf_transitions[:, T.LEFT_ARC.value].view(-1, 1)
        right_arc = clf_transitions[:, T.RIGHT_ARC.value].view(-1, 1)
        shift = clf_transitions[:, T.SHIFT.value].view(-1, 1)
        swap = clf_transitions[:, T.SWAP.value].view(-1, 1)

        # Isolate the columns for the different labels
        relation_slices = self.model.relations.slices
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
        left_arc_scores_sorted, left_arc_scores_indices = \
            torch.sort(left_arc_scores_batch, descending=True)
        right_arc_scores_sorted, right_arc_scores_indices = \
            torch.sort(right_arc_scores_batch, descending=True)

        # Only take the best two items
        left_arc_scores_sorted = left_arc_scores_sorted[:, :2]
        right_arc_scores_sorted = right_arc_scores_sorted[:, :2]

        # We need them later in RAM, so retrieve them all at once from the gpu
        left_arc_scores_indices = left_arc_scores_indices[:, :2].cpu().numpy()
        right_arc_scores_indices = right_arc_scores_indices[:, :2].cpu().numpy()

        combinations = zip(
            configurations, shift_score_batch, swap_score_batch,
            left_arc_scores_batch, right_arc_scores_batch,
            left_arc_scores_sorted, left_arc_scores_indices,
            right_arc_scores_sorted, right_arc_scores_indices
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
