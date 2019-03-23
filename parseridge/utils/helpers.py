import numpy as np
import os
from collections import namedtuple
from enum import Enum

import torch

"""
Container to store a relation that is bound to a transition. 
"""
Relation = namedtuple(
    "Relation",
    [
        "transition",
        "label"
    ]
)

"""
Container that represents a proposed action that the parser could perform.
"""
Action = namedtuple(
    "Action",
    [
        "relation",
        "transition",
        "score"
    ]
)


class Metric:

    def __init__(self, loss=0.0, num_updates=0, iterations=1, num_transitions=0,
                 num_errors=0):
        self.loss = loss
        self.num_updates = num_updates
        self.iterations = iterations
        self.num_transitions = num_transitions
        self.num_errors = num_errors

    def __add__(self, other):
        return Metric(
            self.loss + other.loss,
            self.num_updates + other.num_updates,
            self.iterations + other.iterations,
            self.num_transitions + other.num_transitions,
            self.num_errors + other.num_errors
        )


class Transition(Enum):
    """
    Enumeration class to represent the different transitions in a more readable
    way.
    """
    LEFT_ARC = 0
    RIGHT_ARC = 1
    SHIFT = 2
    SWAP = 3


T = Transition  # Shortcut to make the code less verbose


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


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

def num_same_params(parameters1, parameters2):
    cnt = 0
    for p1, p2 in zip(parameters1, parameters2):
        cnt += int(p1 == p2)
    return cnt
