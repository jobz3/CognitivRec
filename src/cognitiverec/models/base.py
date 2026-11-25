"""The interface every recommender implements. Shared."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


class Recommender(ABC):
    """A model the evaluation harness can drive.

    Models never mask candidates. The harness does that in one place, so every model is ranked
    over the same candidate items.
    """

    @abstractmethod
    def fit(self, history: pd.DataFrame, n_users: int, n_items: int) -> Recommender:
        """Learn from `history` (columns user, item, rating, timestamp; chronological per user)."""

    @abstractmethod
    def score(self, user: int) -> np.ndarray:
        """Return one score per catalogue item, shape (n_items,). Must not change model state."""

    def observe(self, user: int, item: int, rating: float, timestamp: float) -> None:  # noqa: B027
        """Take in a new interaction. Models that cannot adapt ignore it."""

    def record_impression(self, items: np.ndarray) -> None:  # noqa: B027
        """Note that `items` were shown once, as one recommendation list."""


@runtime_checkable
class Explainable(Protocol):
    """A model whose scores can be traced back to individual pieces of user evidence."""

    def attributions(self, user: int, item: int) -> np.ndarray:
        """Return the weight the model gives each evidence entry when scoring `item`."""

    def evidence_items(self, user: int) -> np.ndarray:
        """Return the item ids in attribution and deletion order."""

    def score_without(self, user: int, entry: int) -> np.ndarray:
        """Return full-catalogue scores with evidence entry `entry` removed."""
