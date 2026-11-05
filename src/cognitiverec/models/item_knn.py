"""Model B1: ItemKNN with shrunk cosine similarity (Ferrari Dacrema et al., 2019). Owner: Jobin."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

from cognitiverec.models.base import Recommender


class ItemKNN(Recommender):
    """Score a candidate by summing the user's ratings over the candidate's nearest items.

    A score is exactly the sum of per-history-item contributions, so removing a history item
    lowers the score by exactly its attribution. That makes this model a strong comparison for
    the deletion tests in RQ3.
    """

    def __init__(
        self,
        top_k: int = 80,
        shrinkage: float = 15.0,
        binary: bool = False,
        tfidf: bool = False,
    ) -> None:
        self.top_k = top_k
        self.shrinkage = shrinkage
        self.binary = binary
        self.tfidf = tfidf

    def fit(self, history: pd.DataFrame, n_users: int, n_items: int) -> ItemKNN:
        users = history["user"].to_numpy()
        items = history["item"].to_numpy()
        values = np.ones(len(history)) if self.binary else history["rating"].to_numpy(dtype=float)
        ratings = csr_matrix((values, (users, items)), shape=(n_users, n_items), dtype=np.float64)
        presence = csr_matrix(
            (np.ones(len(history)), (users, items)), shape=(n_users, n_items), dtype=np.float64
        )

        weighted = ratings
        if self.tfidf:
            rated_per_user = np.asarray(presence.sum(axis=1)).ravel()
            weighted = diags(np.log(n_items / (1.0 + rated_per_user))) @ ratings

        norms = np.sqrt(np.asarray(weighted.power(2).sum(axis=0)).ravel()) + 1e-8
        normalised = weighted @ diags(1.0 / norms)
        similarity = (normalised.T @ normalised).toarray()

        co_counts = (presence.T @ presence).toarray()
        denominator = co_counts + self.shrinkage
        denominator[denominator == 0] = 1.0
        similarity *= co_counts / denominator
        np.fill_diagonal(similarity, 0.0)

        # Keep only the top_k neighbours of each candidate item (each column).
        if self.top_k < n_items:
            discard = np.argsort(similarity, axis=0)[: -self.top_k, :]
            np.put_along_axis(similarity, discard, 0.0, axis=0)

        # Row i of neighbours_of holds the weights that history items contribute to candidate i.
        self.neighbours_of = csr_matrix(similarity.T)
        self.profiles = ratings.toarray()
        return self

    def score(self, user: int) -> np.ndarray:
        return self.neighbours_of @ self.profiles[user]

    def observe(self, user: int, item: int, rating: float, timestamp: float) -> None:
        """Fold the interaction into the user's profile. Similarities stay as fitted."""
        self.profiles[user, item] = 1.0 if self.binary else rating

    def _entries(self, user: int) -> np.ndarray:
        return np.flatnonzero(self.profiles[user])

    def attributions(self, user: int, item: int) -> np.ndarray:
        """Return each history item's contribution to the score of `item`."""
        entries = self._entries(user)
        weights = self.neighbours_of[item].toarray().ravel()
        return self.profiles[user, entries] * weights[entries]

    def score_without(self, user: int, entry: int) -> np.ndarray:
        profile = self.profiles[user].copy()
        profile[self._entries(user)[entry]] = 0.0
        return self.neighbours_of @ profile
