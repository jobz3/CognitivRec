"""The exposure-count exploration policy shared by Model D and the factorial study."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cognitiverec.models.base import Recommender


def ucb_bonus(counts: np.ndarray, total: float, beta: float) -> np.ndarray:
    """Return beta * sqrt(ln(T + 1) / (N_exp(i) + 1)) for every item."""
    return beta * np.sqrt(np.log(total + 1.0) / (counts + 1.0))


class ExposureTracker:
    """Global item exposure, starting from interaction counts in the fitting history."""

    def __init__(self, initial_counts: np.ndarray) -> None:
        self.counts = np.asarray(initial_counts, dtype=np.float64).copy()
        self.total = float(self.counts.sum())

    def record(self, items: np.ndarray) -> None:
        """Count one recommendation list: each listed item gains exactly one exposure."""
        self.counts[items] += 1.0
        self.total += len(items)


class ExposureExploration(Recommender):
    """Apply the same exposure bonus to any fitted scorer without changing its preferences."""

    def __init__(self, scorer: Recommender, counts: np.ndarray, beta: float) -> None:
        self.scorer = scorer
        self.beta = beta
        self.exposure = ExposureTracker(counts)

    def fit(self, history: pd.DataFrame, n_users: int, n_items: int) -> ExposureExploration:
        self.scorer.fit(history, n_users, n_items)
        self.exposure = ExposureTracker(np.bincount(history["item"].to_numpy(), minlength=n_items))
        return self

    def score(self, user: int) -> np.ndarray:
        scores = self.scorer.score(user)
        if self.beta == 0:
            return scores
        return scores + ucb_bonus(self.exposure.counts, self.exposure.total, self.beta)

    def observe(self, user: int, item: int, rating: float, timestamp: float) -> None:
        self.scorer.observe(user, item, rating, timestamp)

    def record_impression(self, items: np.ndarray) -> None:
        self.exposure.record(items)
