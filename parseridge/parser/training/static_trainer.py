from torch.utils.data import DataLoader

from parseridge.corpus.training_data import ConLLDataset
from parseridge.parser.loss import Criterion
from parseridge.parser.training.base_trainer import Trainer
from parseridge.parser.training.callbacks.base_callback import StopEpoch, StopTraining
from parseridge.parser.training.hyperparameters import Hyperparameters


class StaticTrainer(Trainer):
    """
    This trainer uses pre-generated training samples.
    """

    def fit(
        self,
        epochs: int,
        training_data: ConLLDataset,
        hyper_parameters: Hyperparameters = None,
        **kwargs,
    ) -> None:
        if not isinstance(training_data, ConLLDataset):
            raise ValueError(
                f"The StaticTrainer requires a ConLLDataset object for training, but "
                f"received a {type(training_data)} object."
            )

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
        self, epoch: int, training_data: ConLLDataset, hyper_parameters: Hyperparameters
    ):
        train_dataloader = DataLoader(
            dataset=training_data,
            batch_size=hyper_parameters.batch_size,
            shuffle=True,
            collate_fn=ConLLDataset.collate_batch,
        )

        num_batches = int(len(training_data) / hyper_parameters.batch_size)

        self.callback_handler.on_epoch_begin(
            epoch=epoch, num_batches=num_batches, training_data=training_data
        )

        criterion = Criterion(loss_function=hyper_parameters.loss_function)

        epoch_loss = 0

        for i, batch_data in enumerate(train_dataloader):
            try:
                self.callback_handler.on_batch_begin(batch=i, batch_data=batch_data)

                batch = ConLLDataset.TrainingBatch(*batch_data)

                pred_transitions, pred_relations = self.model(
                    stacks=batch.stacks,
                    stack_lengths=batch.stack_lengths,
                    buffers=batch.buffers,
                    buffer_lengths=batch.buffer_lengths,
                    token_sequences=batch.sentences,
                    sentence_lengths=batch.sentence_lengths,
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
                    wrong_relations_lengths=batch.wrong_relations_lengths,
                )

                self.learn(loss)

                loss = loss.item()
                epoch_loss += loss

                self.last_epoch = epoch

                self.callback_handler.on_batch_end(
                    batch=i, batch_data=batch_data, batch_loss=loss
                )
            except StopEpoch:
                self.logger.info(f"Stopping epoch after {i}/{num_batches} batches.")
                break

        self.callback_handler.on_epoch_end(epoch=epoch, epoch_loss=epoch_loss)
