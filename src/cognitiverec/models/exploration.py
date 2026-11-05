"""Model D's exploration layer: exposure counts and the UCB bonus. Owner: Abhinav."""

from __future__ import annotations

import numpy as np


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
