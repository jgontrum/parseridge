import os
import random
from collections import namedtuple
from enum import Enum
from typing import Optional

import numpy as np
import torch

from parseridge.utils.logger import LoggerMixin

"""
Container to store a relation that is bound to a transition.
"""
Relation = namedtuple("Relation", ["transition", "label"])


class Metric(LoggerMixin):
    def __init__(
        self,
        loss=0.0,
        num_updates=0,
        iterations=1,
        num_transitions=0,
        num_errors=0,
        num_backprop=0,
    ):
        self.loss = loss
        self.num_updates = num_updates
        self.iterations = iterations
        self.num_transitions = num_transitions
        self.num_errors = num_errors
        self.num_backprop = num_backprop

    def __add__(self, other):
        return Metric(
            self.loss + other.loss,
            self.num_updates + other.num_updates,
            self.iterations + other.iterations,
            self.num_transitions + other.num_transitions,
            self.num_errors + other.num_errors,
            self.num_backprop + other.num_backprop,
        )


class Transition(Enum):
    """
    Enumeration class to represent the different transitions in a more readable
    way.
    """

    LEFT_ARC = 2
    RIGHT_ARC = 3
    SHIFT = 0
    SWAP = 1

    @staticmethod
    def get_item(id_: int) -> str:
        return ["SHIFT", "SWAP", "LEFT_ARC", "RIGHT_ARC"][id_]


T = Transition  # Shortcut to make the code less verbose


class Action:
    """
    Container that represents a proposed action that the parser could perform.
    """

    def __init__(
        self,
        relation: Optional[Relation],
        transition: Transition,
        score: torch.tensor,
        np_score: np.array = None,
    ):
        self.relation = relation
        self.transition = transition
        self.score = score
        self.np_score = np_score

        if isinstance(self.score, torch.Tensor):
            self.np_score = self.score.item()

    @classmethod
    def get_negative_action(cls):
        return cls(None, None, None, np.NINF)

    def get_relation_object(self):
        return Relation(self.transition, self.relation)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def get_parameters(model):
    parameters = []
    for p in model.parameters():
        parameters.append(p.detach().cpu().numpy().flatten())
    return np.concatenate(parameters)


def set_seed(seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        if torch.cuda.device_count() > 0:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(seed)


class RobustDict(dict):
    def __getitem__(self, item):
        return self.get(item)
