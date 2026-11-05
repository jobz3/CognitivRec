"""Model B2: truncated SVD, which also supplies CognitiveRec's item embeddings. Owner: Jobin."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from cognitiverec.models.base import Recommender


class SVD(Recommender):
    """Matrix factorisation of the user-mean-centred rating matrix."""

    def __init__(self, n_factors: int = 32, seed: int = 42) -> None:
        self.n_factors = n_factors
        self.seed = seed

    def fit(self, history: pd.DataFrame, n_users: int, n_items: int) -> SVD:
        profiles = np.zeros((n_users, n_items), dtype=np.float64)
        profiles[history["user"].to_numpy(), history["item"].to_numpy()] = history["rating"]
        rated = profiles > 0
        self.user_means = profiles.sum(axis=1) / np.maximum(1, rated.sum(axis=1))
        centred = profiles - self.user_means[:, None] * rated

        k = min(self.n_factors, min(n_users, n_items) - 1)
        # An explicit start vector keeps the result reproducible across SciPy versions.
        start = np.random.default_rng(self.seed).standard_normal(min(n_users, n_items))
        left, singular, right = svds(csr_matrix(centred), k=k, v0=start)
        order = np.argsort(singular)[::-1]
        left, singular, right = left[:, order], singular[order], right[order]

        self.profiles = profiles
        self.singular = singular
        self.user_vecs = left * np.sqrt(singular)
        self.item_vecs = right.T * np.sqrt(singular)
        return self

    def score(self, user: int) -> np.ndarray:
        return self.item_vecs @ self.user_vecs[user] + self.user_means[user]

    def observe(self, user: int, item: int, rating: float, timestamp: float) -> None:
        """Fold the interaction in by re-projecting the user onto the fitted item factors."""
        self.profiles[user, item] = rating
        profile = self.profiles[user]
        rated = profile > 0
        mean = profile[rated].mean()
        self.user_means[user] = mean
        centred = np.where(rated, profile - mean, 0.0)
        self.user_vecs[user] = centred @ self.item_vecs / np.maximum(self.singular, 1e-12)
