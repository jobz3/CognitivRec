"""A single exponentially weighted latent profile matched to Model C's item representation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cognitiverec.models.base import Recommender
from cognitiverec.models.latent import rating_to_reward, unit_rows, unit_vector
from cognitiverec.models.svd import SVD


class RecencyProfile(Recommender):
    """Weight all feedback by interaction age and score via one candidate-independent profile."""

    def __init__(
        self,
        n_factors: int,
        normalize_embeddings: bool,
        half_life: float | None = None,
        seed: int = 42,
    ) -> None:
        if half_life is not None and (not np.isfinite(half_life) or half_life <= 0):
            raise ValueError("Recency half-life must be finite and positive.")
        self.n_factors = n_factors
        self.normalize_embeddings = normalize_embeddings
        self.half_life = half_life
        self.seed = seed

    def fit(self, history: pd.DataFrame, n_users: int, n_items: int) -> RecencyProfile:
        raw = SVD(n_factors=self.n_factors, seed=self.seed).fit(history, n_users, n_items).item_vecs
        self.unit = unit_rows(raw)
        self.candidates = self.unit if self.normalize_embeddings else raw
        self.profiles = np.zeros((n_users, raw.shape[1]))
        self.decay = 1.0 if self.half_life is None else 2 ** (-1.0 / self.half_life)
        for user, rows in history.groupby("user", sort=False):
            ratings = rating_to_reward(rows["rating"].to_numpy())
            lags = np.arange(len(rows) - 1, -1, -1)
            self.profiles[user] = (ratings * self.decay**lags) @ self.unit[rows["item"].to_numpy()]
        return self

    def score(self, user: int) -> np.ndarray:
        return self.candidates @ unit_vector(self.profiles[user])

    def observe(self, user: int, item: int, rating: float, timestamp: float) -> None:
        self.profiles[user] *= self.decay
        self.profiles[user] += rating_to_reward(rating) * self.unit[item]
