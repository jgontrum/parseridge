import os
from datetime import datetime
from time import time

import torch
from tqdm import tqdm

from parseridge.parser.configuration import Configuration
from parseridge.parser.model import ParseridgeModel
from parseridge.utils.evaluate import CoNNLEvaluator
from parseridge.utils.helpers import T, Metric
from parseridge.utils.logger import LoggerMixin


class ParseRidge(LoggerMixin):
    """
    The core engine of the parser. Here, the training and prediction parts
    are implemented.
    """

    def __init__(self, device):
        self.model = None
        self.model_dir = "./models"
        self.time_prefix = lambda: datetime.now().strftime("%Y%m%d-%H%M%S")
        self.device = device

    def predict(self, corpus, batch_size=512, remove_pbar=True):
        self.model = self.model.eval()
        gold_sentences = []
        pred_sentences = []

        iterator = corpus.get_iterator(batch_size=batch_size, shuffle=False)
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
        return pred_sentences, gold_sentences

    def _run_prediction_batch(self, batch):
        pred_sentences = []
        gold_sentences = []

        sentence_features, sentences = batch

        contextualized_tokens_batch, hidden = self.model.compute_lstm_output(
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

    def fit(self, corpus, relations, batch_size=4, error_prob=0.1, dropout=0.33,
            num_epochs=3, dev_corpus=None):

        self.model = ParseridgeModel(
            relations=relations,
            vocabulary=corpus.vocabulary,
            dropout=dropout,
            device=self.device
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        optimizer.zero_grad()
        self.model.optimizer = optimizer

        evaluator = CoNNLEvaluator()

        for epoch in range(num_epochs):
            t0 = time()
            self.logger.info(f"Starting epoch #{epoch + 1}...")
            epoch_metric = self._run_epoch(corpus, batch_size, error_prob)

            avg_loss = epoch_metric.loss / epoch_metric.num_updates
            self.logger.info(f"Epoch loss: {avg_loss:.8f}")

            scores = {}
            # Evaluate on training corpus
            train_scores = evaluator.get_las_score_for_sentences(
                *self.predict(corpus))
            scores["train"] = train_scores
            self.logger.info(
                f"Performance on the training set after {epoch + 1} epochs: "
                f"LAS: {train_scores['LAS']:.2f} | "
                f"UAS: {train_scores['UAS']:.2f}"
            )

            # Evaluate on dev corpus
            if dev_corpus:
                dev_scores = evaluator.get_las_score_for_sentences(
                    *self.predict(dev_corpus))
                scores["dev"] = dev_scores
                self.logger.info(
                    f"Performance on the dev set after {epoch + 1} epochs: "
                    "     "
                    f"LAS: {dev_scores['LAS']:.2f} | "
                    f"UAS: {dev_scores['UAS']:.2f}"
                )

            path = self.save_model(self.model, scores)
            self.logger.info(f"Saved model to '{path}'.")

            duration = time() - t0
            self.logger.info(
                f"Finished epoch in "
                f"{int(duration / 60)}:{int(duration % 60):01} minutes.")

    def _run_epoch(self, corpus, batch_size=4, error_prob=0.1,
                   update_pbar_interval=50):
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
        loss = []

        epoch_metric = Metric()
        interval_metric = Metric()

        iterator = corpus.get_iterator(batch_size=batch_size, shuffle=True)
        with tqdm(total=len(corpus)) as pbar:
            pbar_template = (
                "Batch Loss: {loss:8.4f} | Updates: {updates:5.1f} | "
                "Induced Errors: {errors:3.1f}"
            )
            pbar.set_description(pbar_template.format(
                loss=0, updates=0, errors=0
            ))
            for i, batch in enumerate(iterator):
                loss, batch_metric = self._run_training_batch(
                    batch, loss, error_prob)

                epoch_metric += batch_metric
                interval_metric += batch_metric

                num_sentences = interval_metric.iterations * batch_size
                if num_sentences >= update_pbar_interval:
                    # Update less frequently

                    desc = pbar_template.format(
                        loss=interval_metric.loss / interval_metric.num_updates,
                        updates=interval_metric.num_updates / num_sentences,
                        errors=interval_metric.num_errors / num_sentences,
                    )
                    pbar.set_description(desc)
                    interval_metric = Metric()

                pbar.update(len(batch[1]))

        # In case there are some margin losses left, compute their gradients.
        self.model.perform_back_propagation(loss)
        return epoch_metric

    def _run_training_batch(self, batch, loss, error_prob):
        """
        Trains the parser model on the data given in `batch` and performs
        back-propagation, if the number of updates is above a certain
        threshold.

        Parameters
        ----------
        batch : tuple of tensor and list of Sentence
            A slice of the training data to train on.
        optimizer : torch.optim object
            The optimizer for the parser model `self.model`.
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
        contextualized_tokens_batch, hidden = self.model.compute_lstm_output(
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
                    configuration.select_actions(actions, costs, error_prob)

                # Apply the dynamic oracle to update the sentence structure
                # for the case that the chosen action does not exactly
                # follow the gold tree.
                configuration.update_dynamic_oracle(best_action, shift_case)

                # Apply the best action and update the stack and buffer
                configuration.apply_transition(best_action)

                # Compute the loss by using the margin between the scores
                if best_valid_action.score < best_wrong_action.score + 1.0:
                    margin = best_wrong_action.score - best_valid_action.score
                    batch_metric.num_updates += 1
                    loss.append(margin)

                if best_action.transition in [T.LEFT_ARC, T.RIGHT_ARC]:
                    batch_metric.num_transitions += 1
                if best_action != best_valid_action:
                    batch_metric.num_errors += 1

            # Remove all finished configurations
            configurations = [c for c in configurations if not c.is_terminal]

        # Perform back propagation
        loss, stats = self.model.perform_back_propagation(loss)
        batch_metric.loss += stats

        return loss, batch_metric

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
            [c.buffer for c in configurations],
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

    def save_model(self, model, scores):
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        current_path = f"{self.model_dir}/model-{self.time_prefix()}.model"

        torch.save({
            "model": model.state_dict(),
            "vocabulary": model.vocabulary,
            "relations": model.relations,
            "scores": scores
        }, current_path)

        return current_path

    def load_model(self, path):
        model_data = torch.load(path)

        self.model = ParseridgeModel(
            model_data["relations"],
            model_data["vocabulary"],
            device=self.device
        )
        self.model.load_state_dict(model_data["model"])
        self.model = self.model.to(self.device)
