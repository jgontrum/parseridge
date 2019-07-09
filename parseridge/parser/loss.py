from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss

from parseridge.utils.logger import LoggerMixin


class Loss(ABC, LoggerMixin):
    @abstractmethod
    def __call__(
        self,
        pred_transitions: Tensor,
        pred_relations: Tensor,
        gold_transitions: Tensor,
        gold_relations: Tensor,
        wrong_transitions: Tensor,
        wrong_transitions_lengths: Tensor,
        wrong_relations: Tensor,
        wrong_relations_lengths: Tensor,
    ) -> Tensor:
        raise NotImplementedError()


class PyTorchLoss(Loss):
    """
    Wrapper that calls the given PyTorch loss function and strips all other arguments.
    """

    def __init__(self, criterion):
        self.criterion = criterion
        assert isinstance(self.criterion, _Loss)

    def __call__(
        self,
        pred_transitions: Tensor,
        pred_relations: Tensor,
        gold_transitions: Tensor,
        gold_relations: Tensor,
        **kwargs,
    ) -> Tensor:
        loss_transition = self.criterion(pred_transitions, gold_transitions)
        loss_relation = self.criterion(pred_relations, gold_relations)

        return loss_transition + loss_relation


class MaxMarginLoss(Loss):
    def __init__(self, margin_threshold: float = 1.0):
        self.margin_threshold = margin_threshold

    def __call__(
        self,
        pred_transitions: Tensor,
        pred_relations: Tensor,
        gold_transitions: Tensor,
        gold_relations: Tensor,
        wrong_transitions: Tensor,
        wrong_transitions_lengths: Tensor,
        wrong_relations: Tensor,
        wrong_relations_lengths: Tensor,
    ) -> Tensor:
        gold_scores = self._get_gold_scores(
            pred_transitions, pred_relations, gold_transitions, gold_relations
        )

        wrong_scores = self._get_wrong_scores(
            pred_transitions,
            pred_relations,
            wrong_transitions,
            wrong_relations,
            wrong_transitions_lengths,
        )

        # Compute the margin between the best wrong scores and the gold scores
        scores = wrong_scores - gold_scores

        # Sum the loss only for those items where the difference is below the threshold
        masked_scores = scores[gold_scores < wrong_scores + self.margin_threshold]
        loss = torch.mean(masked_scores)

        return loss

    @staticmethod
    def _get_gold_scores(
        pred_transitions: Tensor,
        pred_relations: Tensor,
        gold_transitions: Tensor,
        gold_relations: Tensor,
    ) -> Tensor:
        # Compute the scores of the gold items by adding the score for the relation to
        # the score of the transition.
        gold_transitions_scores = pred_transitions.gather(
            dim=1, index=gold_transitions.unsqueeze(1)
        )

        gold_relations_scores = pred_relations.gather(
            dim=1, index=gold_relations.unsqueeze(1)
        )

        return (gold_transitions_scores + gold_relations_scores).squeeze()

    @staticmethod
    def _get_wrong_scores(
        pred_transitions: Tensor,
        pred_relations: Tensor,
        wrong_transitions: Tensor,
        wrong_relations: Tensor,
        wrong_transitions_lengths: Tensor,
    ) -> Tensor:
        # In every batch, compute a score for all the wrong items
        wrong_transitions_scores = torch.gather(pred_transitions, 1, wrong_transitions)
        wrong_relations_scores = torch.gather(pred_relations, 1, wrong_relations)

        wrong_scores = wrong_transitions_scores + wrong_relations_scores

        # For clarity, we rename the variable,
        # since wrong_transitions_lengths ==  wrong_relations_lengths
        wrong_actions_lengths = wrong_transitions_lengths

        # Create a mask based on sequence lengths.
        # See: http://juditacs.github.io/2018/12/27/masked-attention.html
        max_len = wrong_scores.size(1)
        device = wrong_actions_lengths.device
        mask = (
            torch.arange(max_len, device=device)[None, :] < wrong_actions_lengths[:, None]
        )

        # Invert mask and blank out all padding.
        wrong_scores[~mask] = float("-inf")

        # Get the best wrong action for each item in the batch
        wrong_scores, _ = torch.max(wrong_scores, dim=1)

        return wrong_scores


class Criterion(Loss):
    LOSS_FUNCTIONS = {
        "CrossEntropy": lambda kwargs: PyTorchLoss(nn.CrossEntropyLoss(**kwargs)),
        "MSELoss": lambda kwargs: PyTorchLoss(nn.MSELoss(**kwargs)),
        "NLLLoss": lambda kwargs: PyTorchLoss(nn.NLLLoss(**kwargs)),
        "MaxMargin": lambda kwargs: MaxMarginLoss(**kwargs),
    }

    def __init__(self, loss_function: str = "CrossEntropy", **kwargs):
        if loss_function not in self.LOSS_FUNCTIONS:
            raise ValueError(
                f"Unknown loss function: {loss_function}. "
                f"Must be one of {list(self.LOSS_FUNCTIONS.keys())}"
            )

        self.criterion = self.LOSS_FUNCTIONS[loss_function](kwargs)

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.criterion(*args, **kwargs)
