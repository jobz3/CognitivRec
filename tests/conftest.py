"""Shared fixtures: a small synthetic dataset, so that the offline tests need no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cognitiverec.data import Dataset, chronological_split
from cognitiverec.evaluation.protocol import test_phase, validation_phase
from cognitiverec.models import SVD, CognitiveRecommender, ItemKNN, Popularity


def make_dataset(n_users: int = 40, n_items: int = 60, seed: int = 0) -> Dataset:
    """Two taste groups: odd users like odd items, even users like even items."""
    rng = np.random.default_rng(seed)
    genre_names = ("unknown", "Alpha", "Beta", "Gamma", "Delta")
    item_genres = np.zeros((n_items, len(genre_names)), dtype=np.int8)
    item_genres[np.arange(n_items), 1 + np.arange(n_items) % 4] = 1

    rows = []
    for user in range(n_users):
        items = rng.choice(n_items, size=int(rng.integers(20, 35)), replace=False)
        for position, item in enumerate(items):
            liked = int(item) % 2 == user % 2
            rating = float(rng.choice([4.0, 5.0]) if liked else rng.choice([1.0, 2.0, 3.0]))
            rows.append((user, int(item), rating, 1_000 + position))
    interactions = pd.DataFrame(rows, columns=["user", "item", "rating", "timestamp"])
    return Dataset(
        interactions=interactions,
        item_genres=item_genres,
        item_titles=tuple(f"item {index}" for index in range(n_items)),
        genre_names=genre_names,
        n_users=n_users,
        n_items=n_items,
    )


MODEL_FACTORIES = {
    "popularity": Popularity,
    "item_knn": lambda: ItemKNN(top_k=10, shrinkage=1.0),
    "svd": lambda: SVD(n_factors=4),
    "cognitive": lambda: CognitiveRecommender(n_factors=4, memory_size=4),
    "cognitive_ucb": lambda: CognitiveRecommender(
        n_factors=4, memory_size=4, exploration_beta=0.05
    ),
}


@pytest.fixture(params=sorted(MODEL_FACTORIES))
def factory(request):
    """Run a test once per model type."""
    return MODEL_FACTORIES[request.param]


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    return make_dataset()


@pytest.fixture(scope="session")
def split(dataset):
    return chronological_split(dataset.interactions)


@pytest.fixture(scope="session")
def val_phase(dataset, split):
    return validation_phase(dataset, split)


@pytest.fixture(scope="session")
def final_phase(dataset, split):
    return test_phase(dataset, split)
