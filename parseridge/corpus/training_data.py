import pickle
from copy import deepcopy
from dataclasses import dataclass
from random import random, choice, shuffle
from typing import List

import torch
from torch.utils.data import Dataset as PyTorchDataset
from tqdm.auto import tqdm

from parseridge.corpus.relations import Relations
from parseridge.corpus.sentence import Sentence
from parseridge.corpus.vocabulary import Vocabulary
from parseridge.parser.configuration import Configuration
from parseridge.parser.modules.utils import pad_tensor
from parseridge.utils.helpers import Action, T
from parseridge.utils.logger import LoggerMixin


@dataclass
class ConfigurationItem:
    sentence: torch.Tensor
    stack: torch.Tensor
    buffer: torch.Tensor
    gold_transition: torch.Tensor
    gold_relation: torch.Tensor
    wrong_transitions: torch.Tensor
    wrong_relations: torch.Tensor

    _current_iteration: int = 0

    def to(self, device) -> "ConfigurationItem":
        new_item = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                new_item[k] = v.to(device)

        return ConfigurationItem(**new_item)

    def __iter__(self):
        return self

    def __next__(self):
        fields = [v for k, v in self.__dict__.items() if not k.startswith("_")]
        if self._current_iteration >= len(fields):
            self._current_iteration = 0
            raise StopIteration
        else:
            self._current_iteration += 1
            return fields[self._current_iteration - 1]


@dataclass
class ConLLDataset(PyTorchDataset, LoggerMixin):
    data_points: List[ConfigurationItem]
    vocabulary: Vocabulary
    relations: Relations
    device: str = "cpu"

    def to(self, device: str) -> "ConLLDataset":
        data_points = [data_point.to(device) for data_point in self.data_points]
        return ConLLDataset(data_points, self.vocabulary, self.relations, device)

    def __len__(self):
        return len(self.data_points)

    def get_length_tensor(self, tensor):
        if not len(tensor.shape):
            # Handling scalars
            return torch.tensor(1, device=self.device)

        return torch.tensor(tensor.shape[0], device=self.device)

    def __getitem__(self, index: int):
        if not self.data_points:
            raise ValueError("Call `ConLLDataset.reset()` first to generate data points.")

        item = self.data_points[index]

        features = []
        for feature in item:
            features.append(feature)
            features.append(self.get_length_tensor(feature))

        return tuple(features)

    @staticmethod
    def collate_batch(batch):
        """
        Collate function that must be used in the DataLoader for this DataSet.
        Pads the sequences of various length and returns a list of tensors.

        Parameters
        ----------
        batch : list of tuples of Tensor

        Returns
        -------
        List of Tensor
        """
        # Sort batch according to sentence length
        sentence_lengths = [item[1] for item in batch]
        order = sorted(
            range(len(sentence_lengths)), key=lambda k: sentence_lengths[k], reverse=True
        )
        batch = [batch[i] for i in order]

        ret = []
        num_features = len(batch[0])
        for idx in range(0, num_features, 2):
            lengths = [item[idx + 1] for item in batch]

            if not len(batch[0][idx].shape):
                # Don't pad scalars
                features = torch.stack([item[idx] for item in batch])
            else:
                max_length = max(lengths)
                features = torch.stack(
                    [pad_tensor(item[idx], max_length, padding=0) for item in batch]
                )

            ret.append(features)
            ret.append(torch.stack(lengths))

        return ret

    def save(self, path: str):
        pickle.dump(self.to("cpu"), open(path, "wb"), protocol=4)

    @classmethod
    def load(cls, path: str):
        return pickle.load(open(path, "rb"))

    @dataclass
    class TrainingBatch:
        sentences: torch.Tensor
        sentence_lengths: torch.Tensor
        stacks: torch.Tensor
        stack_lengths: torch.Tensor
        buffers: torch.Tensor
        buffer_lengths: torch.Tensor
        gold_transitions: torch.Tensor
        gold_transitions_lengths: torch.Tensor
        gold_relations: torch.Tensor
        gold_relations_lengths: torch.Tensor
        wrong_transitions: torch.Tensor
        wrong_transitions_lengths: torch.Tensor
        wrong_relations: torch.Tensor
        wrong_relations_lengths: torch.Tensor


@dataclass
class DataGenerator(LoggerMixin):
    vocabulary: Vocabulary
    relations: Relations
    oov_probability: float = 0.25
    token_dropout: float = 0.001
    error_probability: float = 0.1
    device: str = "cpu"

    def generate_configuration(self, sentence: Sentence) -> ConfigurationItem:
        configuration = Configuration(
            sentence, contextualized_input=None, model=None, device=self.device
        )

        item_filter = set()

        for configuration_item in self._generate_next_datapoint(configuration):
            feature_key = (
                tuple(configuration_item.stack.cpu().tolist()),
                tuple(configuration_item.buffer.cpu().tolist()),
            )

            if feature_key not in item_filter:
                item_filter.add(feature_key)
                yield configuration_item

    def generate_dataset(self, sentences: List[Sentence]) -> ConLLDataset:
        new_data_points = []
        with tqdm(sentences, desc="Generating training examples", maxinterval=1) as pbar:
            for sentence in pbar:
                new_data_points += list(self.generate_configuration(sentence))
                pbar.set_description(
                    f"Generating training examples ({len(new_data_points)})"
                )

        return ConLLDataset(
            data_points=new_data_points,
            vocabulary=self.vocabulary,
            relations=self.relations,
            device=self.device,
        )

    def _generate_next_datapoint(self, configuration):
        if not configuration.is_terminal:
            stack = configuration.stack_tensor
            buffer = configuration.buffer_tensor

            possible_actions = list(self._get_possible_action(configuration))
            costs, shift_case = configuration.get_transition_costs(possible_actions)

            valid_actions = configuration.get_valid_actions(possible_actions, costs)
            wrong_actions = configuration.get_wrong_actions(possible_actions, costs)

            if valid_actions:
                actions = [("valid", choice(valid_actions))]

                if random() < self.error_probability and costs[T.SWAP] != 0:
                    selected_wrong_actions = self._remove_label_duplicates(wrong_actions)

                    transitions = set([a.transition for a in valid_actions])
                    selected_wrong_actions = [
                        a
                        for a in selected_wrong_actions
                        if a.transition != T.SWAP and a.transition not in transitions
                    ]

                    if selected_wrong_actions:
                        wrong_action = choice(selected_wrong_actions)
                        actions.append(("wrong", wrong_action))

                shuffle(actions)
                for i, (source, action) in enumerate(actions):
                    if len(actions) == 1 or i == len(actions) - 1:
                        # If this the only / last action, reuse the existing
                        # configuration to avoid the deepcopy overhead.
                        new_config = configuration
                    else:
                        new_config = Configuration(
                            deepcopy(configuration.sentence),
                            None,
                            None,
                            False,
                            configuration.device,
                        )
                        new_config.buffer = deepcopy(configuration.buffer)
                        new_config.stack = deepcopy(configuration.stack)

                    new_config.update_dynamic_oracle(action, shift_case)
                    new_config.apply_transition(action)

                    gold_transition, gold_relation = self._get_gold_labels(action)

                    if source == "valid":
                        wrong_transitions_tensor, wrong_relations_tensor = self._get_all_labels(
                            wrong_actions
                        )

                        yield ConfigurationItem(
                            sentence=self._get_sentence_tensor(new_config.sentence),
                            stack=stack,
                            buffer=buffer,
                            gold_transition=gold_transition,
                            gold_relation=gold_relation,
                            wrong_transitions=wrong_transitions_tensor,
                            wrong_relations=wrong_relations_tensor,
                        )

                    for configuration_item in self._generate_next_datapoint(new_config):
                        yield configuration_item

    @staticmethod
    def _remove_label_duplicates(actions):
        seen_transitions = set()

        filtered_actions = []
        for action in actions:
            if action.transition not in seen_transitions:
                seen_transitions.add(action.transition)
                filtered_actions.append(action)

        return filtered_actions

    @staticmethod
    def _select_actions(actions):
        probabilities = [1.0, 0.4, 0.1]
        filtered_actions = []

        for probability in probabilities:
            if random() <= probability:
                other_actions = [
                    action for action in actions if action not in filtered_actions
                ]
                if not other_actions:
                    break

                filtered_actions.append(choice(other_actions))

        return filtered_actions

    def _get_sentence_tensor(self, sentence):
        tokens = []

        for token in sentence:
            token_id = self.vocabulary.add(token.form)
            frequency = self.vocabulary.get_count(token.form)

            if self.oov_probability:
                if random() > (frequency + (frequency / self.oov_probability)):
                    token_id = self.vocabulary.oov

            if random() < self.token_dropout:
                token_id = self.vocabulary.oov

            tokens.append(token_id)

        return torch.tensor(tokens, dtype=torch.int64, device=self.device)

    def _get_gold_labels(self, action):
        relation_id = self.relations.label_signature.get_id(action.get_relation_object())

        return (
            torch.tensor(action.transition.value, dtype=torch.int64, device=self.device),
            torch.tensor(relation_id, dtype=torch.int64, device=self.device),
        )

    def _get_all_labels(self, actions):
        transitions = []
        relations = []

        for action in actions:
            relation_id = self.relations.label_signature.get_id(
                action.get_relation_object()
            )
            transition_id = action.transition.value

            relations.append(relation_id)
            transitions.append(transition_id)

        return (
            torch.tensor(transitions, dtype=torch.int64, device=self.device),
            torch.tensor(relations, dtype=torch.int64, device=self.device),
        )

    @staticmethod
    def _get_possible_action(configuration):
        if configuration.left_arc_conditions:
            yield Action(
                relation=configuration.top_stack_token.relation,
                transition=T.LEFT_ARC,
                score=1.0,
                np_score=1.0,
            )

        if configuration.right_arc_conditions:
            yield Action(
                relation=configuration.top_stack_token.relation,
                transition=T.RIGHT_ARC,
                score=1.0,
                np_score=1.0,
            )

        if configuration.shift_conditions:
            yield Action(relation=None, transition=T.SHIFT, score=1.0, np_score=1.0)

        if configuration.swap_conditions:
            yield Action(relation=None, transition=T.SWAP, score=1.0, np_score=1.0)
