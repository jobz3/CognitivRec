"""Model A: static global popularity. Owner: Jobin."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cognitiverec.models.base import Recommender


class Popularity(Recommender):
    """Rank items by how many users interacted with them in the fitting history."""

    def fit(self, history: pd.DataFrame, n_users: int, n_items: int) -> Popularity:
        self.counts = np.bincount(history["item"].to_numpy(), minlength=n_items).astype(np.float64)
        return self

    def score(self, user: int) -> np.ndarray:
        return self.counts.copy()
