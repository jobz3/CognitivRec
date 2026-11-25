"""EASE, the closed-form item-to-item ridge model of Steck (2019)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from cognitiverec.models.base import Recommender


class EASE(Recommender):
    """Use positive histories to predict items, with the self-prediction diagonal fixed at zero."""

    def __init__(self, regularization: float = 100.0, positive_threshold: float = 4.0) -> None:
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("EASE regularization must be finite and positive.")
        self.regularization = regularization
        self.positive_threshold = positive_threshold

    def fit(self, history: pd.DataFrame, n_users: int, n_items: int) -> EASE:
        self.profiles = np.zeros((n_users, n_items), dtype=float)
        positive = history.loc[history["rating"] >= self.positive_threshold]
        self.profiles[positive["user"].to_numpy(), positive["item"].to_numpy()] = 1.0
        gram = self.profiles.T @ self.profiles
        gram.flat[:: n_items + 1] += self.regularization
        precision = cho_solve(cho_factor(gram), np.eye(n_items))
        self.weights = -precision / np.diag(precision)
        np.fill_diagonal(self.weights, 0.0)
        return self

    def score(self, user: int) -> np.ndarray:
        items = np.flatnonzero(self.profiles[user])
        return self.weights[items].sum(axis=0)

    def observe(self, user: int, item: int, rating: float, timestamp: float) -> None:
        self.profiles[user, item] = rating >= self.positive_threshold
