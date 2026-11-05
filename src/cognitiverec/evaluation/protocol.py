"""The one evaluation protocol every model goes through. Shared.

Candidate masking, top-k selection and user order happen only here, so all models are compared on
the same candidate items.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cognitiverec.data import Dataset, Split
from cognitiverec.evaluation.quality import ranking_metrics, self_information
from cognitiverec.models.base import Recommender


class Phase:
    """What a model may learn from (`history`) and what it is being asked to predict (`targets`).

    Per-user lookups keep rows in time order. `positives` holds target items rated at or above
    `positive_threshold`, and `users` lists the users who have at least one.
    """

    def __init__(
        self,
        name: str,
        history: pd.DataFrame,
        targets: pd.DataFrame,
        dataset: Dataset,
        positive_threshold: float = 4.0,
    ) -> None:
        self.name = name
        self.history = history
        self.targets = targets
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.item_genres = dataset.item_genres
        self.genre_names = dataset.genre_names
        self.positive_threshold = positive_threshold
        self.popularity = np.bincount(history["item"].to_numpy(), minlength=dataset.n_items)
        self.self_information = self_information(self.popularity, dataset.n_users)
        self.seen = {int(user): group["item"].to_numpy() for user, group in history.groupby("user")}
        self.streams = {int(user): group for user, group in targets.groupby("user")}
        positive = targets[targets["rating"] >= positive_threshold]
        self.positives = {
            int(user): set(group["item"].tolist()) for user, group in positive.groupby("user")
        }
        self.users = np.array(sorted(self.positives), dtype=np.int64)

    def seen_items(self, user: int) -> np.ndarray:
        return self.seen.get(user, np.empty(0, dtype=np.int64))

    def fit(self, model: Recommender) -> Recommender:
        """Fit `model` on this phase's history."""
        return model.fit(self.history, self.n_users, self.n_items)


def validation_phase(dataset: Dataset, split: Split, positive_threshold: float = 4.0) -> Phase:
    """For tuning: learn from train, predict validation."""
    return Phase("val", split.train, split.val, dataset, positive_threshold)


def test_phase(dataset: Dataset, split: Split, positive_threshold: float = 4.0) -> Phase:
    """For final results: learn from train and validation, predict test."""
    history = pd.concat([split.train, split.val]).sort_values("user", kind="stable")
    return Phase("test", history.reset_index(drop=True), split.test, dataset, positive_threshold)


test_phase.__test__ = False  # stop pytest collecting this function as a test


def rank_top_k(scores: np.ndarray, exclude: np.ndarray, k: int) -> np.ndarray:
    """Return the k best candidates, never including `exclude`. Ties go to the lower item id."""
    masked = np.array(scores, dtype=np.float64)
    masked[exclude] = -np.inf
    order = np.argsort(-masked, kind="stable")
    return order[np.isfinite(masked[order])][:k]


def user_order(users: np.ndarray, rng: np.random.Generator | None) -> np.ndarray:
    """Return the order users are served in. Shuffled when `rng` is given, otherwise as is.

    Order matters for models whose state depends on earlier recommendations (Model D's exposure
    counts), so it is seeded rather than always running from the lowest user id.
    """
    return users if rng is None else rng.permutation(users)


@dataclass
class RankingRun:
    """One recommendation list per user, and the per-user quality metrics for those lists."""

    per_user: pd.DataFrame
    recommendations: dict[int, np.ndarray]


def evaluate_ranking(
    model: Recommender, phase: Phase, k: int = 10, rng: np.random.Generator | None = None
) -> RankingRun:
    """Recommend once to each user and score against all of their positive targets.

    Works on a copy, so the caller's model is unchanged and every test starts from the same state.
    """
    model = copy.deepcopy(model)
    rows = []
    recommendations = {}
    for user in user_order(phase.users, rng):
        user = int(user)
        items = rank_top_k(model.score(user), phase.seen_items(user), k)
        model.record_impression(items)
        recommendations[user] = items
        metrics = ranking_metrics(
            items, phase.positives[user], k, phase.self_information, phase.item_genres
        )
        rows.append({"user": user, **metrics})
    per_user = pd.DataFrame(rows).sort_values("user").reset_index(drop=True)
    return RankingRun(per_user, recommendations)
