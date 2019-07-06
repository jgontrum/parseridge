from datetime import datetime
from time import time
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from parseridge.corpus.corpus import CorpusIterator, Corpus
from parseridge.corpus.relations import Relations
from parseridge.corpus.training_data import ConLLDataset
from parseridge.corpus.vocabulary import Vocabulary
from parseridge.parser.configuration import Configuration
from parseridge.parser.loss import Criterion
from parseridge.parser.model import ParseridgeModel
from parseridge.parser.modules.utils import pad_tensor_list
from parseridge.parser.trainer import Trainer
from parseridge.parser.evaluation.conll_eval import CoNLLEvaluationScript
from parseridge.utils.helpers import T, Metric
from parseridge.utils.logger import LoggerMixin
from parseridge.utils.report import get_reporter


class Parseridge(LoggerMixin):
    """
    The core engine of the parser. Here, the training and prediction parts
    are implemented.
    """

    def __init__(self, device):
        self.stats = []
        self.time_prefix = lambda: datetime.now().strftime("%Y%m%d-%H%M%S")

        self.model = None
        self.trainer = None
        self.reporter = None

        self.id_ = str(self.time_prefix())
        self.device = device

    def fit(self, train_sentences, dev_sentences, num_stack=3, num_buffer=1,
            embedding_size=100, lstm_hidden_size=125, lstm_layers=2,
            relation_mlp_layers=None, transition_mlp_layers=None, margin_threshold=2.5,
            error_probability=0.1, oov_probability=0.25, token_dropout=0.01,
            lstm_dropout=0.33, mlp_dropout=0.25, batch_size=4, pred_batch_size=512,
            num_epochs=3, gradient_clipping=10.0, weight_decay=0.0, learning_rate=0.001,
            update_size=50, loss_factor=0.75, loss_strategy="avg", google_sheet_id=None,
            google_sheet_auth_file=None, embeddings=None, loss_function="CrossEntropy",
            params=None):

        # The vocabulary maps tokens to integer ids, while the relations object
        # manages the relation labels and their position in the MLP output.
        vocabulary = Vocabulary(embeddings_vocab=embeddings.vocab if embeddings else None)
        relations = Relations(train_sentences)

        if embeddings:
            if embedding_size != embeddings.dim:
                self.logger.warning(
                    "Overwriting embedding dimensions to match external embeddings.")
                embedding_size = embeddings.dim

        # Generate the training examples based on the dependency graphs in the train data
        self.train_dataset = ConLLDataset(
            train_sentences,
            vocabulary=vocabulary,
            relations=relations,
            oov_probability=oov_probability,
            error_probability=error_probability,
            device=self.device
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ConLLDataset.collate_batch
        )

        relations.signature.read_only()

        # Prepare corpus objects which store information about the whole sentence.
        # They are used for evaluation, not training.
        train_corpus = Corpus(train_sentences, vocabulary, device=self.device)
        dev_corpus = Corpus(dev_sentences, vocabulary, device=self.device)

        vocabulary.read_only()

        self.model = ParseridgeModel(
            relations=relations,
            vocabulary=vocabulary,
            num_stack=num_stack,
            num_buffer=num_buffer,
            lstm_dropout=lstm_dropout,
            mlp_dropout=mlp_dropout,
            embedding_size=embedding_size,
            embeddings=embeddings,
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

        criterion = Criterion(loss_function=loss_function)

        # torch.autograd.set_detect_anomaly(True)

        with get_reporter(
                sheets_id=google_sheet_id,
                auth_file_path=google_sheet_auth_file,
                hyper_parameters=params
        ) as self.reporter:
            for epoch in range(num_epochs):
                t0 = time()
                self.logger.info(f"Starting epoch #{epoch + 1}...")

                epoch_metric = self._run_epoch(
                    dataloader=self.train_dataloader,
                    criterion=criterion,
                    epoch=epoch + 1
                )

                # Evaluate on training corpus
                train_scores = CoNLLEvaluationScript().get_las_score_for_sentences(
                    *self.predict(train_corpus, batch_size=pred_batch_size))
                self.logger.info(
                    f"Performance on the training set after {epoch + 1} epochs: "
                    f"LAS: {train_scores['LAS']:.2f} | "
                    f"UAS: {train_scores['UAS']:.2f}"
                )

                # Evaluate on dev corpus
                dev_scores = CoNLLEvaluationScript().get_las_score_for_sentences(
                    *self.predict(dev_corpus, batch_size=pred_batch_size))

                self.logger.info(
                    f"Performance on the dev set after {epoch + 1} epochs: "
                    "     "
                    f"LAS: {dev_scores['LAS']:.2f} | "
                    f"UAS: {dev_scores['UAS']:.2f}"
                )

                duration = time() - t0

                self.reporter.report_epoch(
                    epoch=epoch + 1,
                    epoch_loss=epoch_metric.loss,
                    train_las=train_scores["LAS"],
                    train_uas=train_scores["UAS"],
                    dev_las=dev_scores["LAS"],
                    dev_uas=dev_scores["UAS"]
                )

                self.logger.info(
                    f"Finished epoch in "
                    f"{int(duration / 60)}:{int(duration % 60):01} minutes.")

    def _run_epoch(self, dataloader, criterion: Criterion, epoch=None):
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
        self.model.before_epoch()

        epoch_metric = Metric()

        loss_values = []

        progress_bar = tqdm(dataloader, total=len(dataloader.dataset), desc="Training")
        for i, batch_tuple in enumerate(dataloader):
            batch = ConLLDataset.TrainingBatch(*batch_tuple)

            self.model.before_batch()

            pred_transitions, pred_relations = self.model(
                sentences=batch.sentences,
                sentence_lengths=batch.sentence_lengths,
                stacks=batch.stacks,
                stack_lengths=batch.stack_lengths,
                buffers=batch.buffers,
                buffer_lengths=batch.buffer_lengths
            )

            # Compute loss. Depending on the chosen loss strategy only a part of the
            # arguments will actually be used in the computations of the loss value.
            loss = criterion(
                pred_transitions=pred_transitions,
                gold_transitions=batch.gold_transitions,
                pred_relations=pred_relations,
                gold_relations=batch.gold_relations,
                wrong_transitions=batch.wrong_transitions,
                wrong_transitions_lengths=batch.wrong_transitions_lengths,
                wrong_relations=batch.wrong_relations,
                wrong_relations_lengths=batch.wrong_relations_lengths
            )

            # Back-propagate
            loss.backward()
            self.trainer.optimizer.step()
            self.trainer.optimizer.zero_grad()

            self.model.after_batch()

            # Log the loss for analytics
            loss_values.append(loss.item())
            if i > 0 and i % 100 == 0 and self.reporter:
                self.reporter.report_loss(
                    loss_value=sum(loss_values),
                    epoch=epoch
                )
                loss_values = []

            epoch_metric += Metric(
                loss=loss.item()
            )

            progress_bar.update(len(batch_tuple[0]))

        progress_bar.close()
        self.model.after_epoch()
        return epoch_metric

    def predict(self, corpus, batch_size=512, remove_pbar=True):
        self.model = self.model.eval()

        gold_sentences = []
        pred_sentences = []

        iterator = CorpusIterator(corpus, batch_size=batch_size, train=False)
        with tqdm(
                total=len(corpus),
                desc="Predicting sentences...",
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

        sentence_features = torch.squeeze(sentence_features, dim=1)

        sentence_lengths = [len(sentence) for sentence in sentences]
        sentence_lengths = torch.tensor(
            sentence_lengths, dtype=torch.int64, device=self.device)

        contextualized_tokens_batch = self.model.compute_lstm_output(
            sentence_features, sentence_lengths
        )

        configurations = [
            Configuration(sentence, contextualized_input, self.model,
                          sentence_features=sentence_feature)
            for contextualized_input, sentence, sentence_feature in
            zip(contextualized_tokens_batch, sentences, sentence_features)
        ]

        while configurations:
            # Pass the stacks and buffers through the MLPs in one batch
            configurations, _, _ = self._update_classification_scores(
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
        stacks = [c.stack_tensor for c in configurations]
        stacks_padded = pad_tensor_list(stacks)
        stacks_lengths = torch.tensor(
            [len(c.stack) for c in configurations],
            dtype=torch.int64, device=self.device
        )

        buffers = [c.buffer_tensor for c in configurations]
        buffers_padded = pad_tensor_list(buffers)
        buffer_lengths = torch.tensor(
            [len(c.buffer) for c in configurations],
            dtype=torch.int64, device=self.device
        )

        clf_transitions, clf_relations = self.model(
            sentences=torch.stack([c.sentence_features for c in configurations]),
            sentence_lengths=None,
            sentence_encoding_batch=[c.contextualized_input for c in configurations],
            buffers=buffers_padded,
            buffer_lengths=buffer_lengths,
            stacks=stacks_padded,
            stack_lengths=stacks_lengths
        )

        # Isolate the columns for the transitions
        left_arc = clf_transitions[:, T.LEFT_ARC.value].view(-1, 1)
        right_arc = clf_transitions[:, T.RIGHT_ARC.value].view(-1, 1)
        shift = clf_transitions[:, T.SHIFT.value].view(-1, 1)
        swap = clf_transitions[:, T.SWAP.value].view(-1, 1)

        # Isolate the columns for the different relations
        relation_slices = self.model.relations.slices
        shift_relations = clf_relations[:, relation_slices[T.SHIFT]]
        swap_relations = clf_relations[:, relation_slices[T.SWAP]]
        left_arc_relations = clf_relations[:, relation_slices[T.LEFT_ARC]]
        right_arc_relations = clf_relations[:, relation_slices[T.RIGHT_ARC]]

        # Add them in one batch
        shift_score_batch = torch.add(shift, shift_relations)
        swap_score_batch = torch.add(swap, swap_relations)
        left_arc_scores_batch = torch.add(left_arc, left_arc_relations)
        right_arc_scores_batch = torch.add(right_arc, right_arc_relations)

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

        class Combination(NamedTuple):
            configuration: Configuration

            shift_score: torch.Tensor
            swap_score: torch.Tensor

            left_arc_scores: torch.Tensor
            left_arc_scores_indices: np.array
            left_arc_scores_sorted: torch.Tensor

            right_arc_scores: torch.Tensor
            right_arc_scores_indices: np.array
            right_arc_scores_sorted: torch.Tensor

            def apply(self):
                self.configuration.scores = {
                    T.SHIFT: self.shift_score,
                    T.SWAP: self.swap_score,
                    T.LEFT_ARC: self.left_arc_scores,
                    (T.LEFT_ARC, "best_scores"): self.left_arc_scores_sorted,
                    (T.LEFT_ARC, "best_scores_indices"): self.left_arc_scores_indices,
                    T.RIGHT_ARC: self.right_arc_scores,
                    (T.RIGHT_ARC, "best_scores"): self.right_arc_scores_sorted,
                    (T.RIGHT_ARC, "best_scores_indices"): self.right_arc_scores_indices
                }

        combinations = zip(
            configurations, shift_score_batch, swap_score_batch,
            left_arc_scores_batch, left_arc_scores_indices, left_arc_scores_sorted,
            right_arc_scores_batch, right_arc_scores_indices, right_arc_scores_sorted
        )

        # Update the result of the classifiers in the configurations
        for combination in combinations:
            Combination(*combination).apply()

        return configurations, clf_transitions, clf_relations
