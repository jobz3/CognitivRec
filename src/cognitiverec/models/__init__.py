"""Recommender models and a factory for building them from configuration."""

from __future__ import annotations

from typing import Any

from cognitiverec.models.base import Explainable, Recommender
from cognitiverec.models.cognitive import CognitiveRecommender
from cognitiverec.models.ease import EASE
from cognitiverec.models.item_knn import ItemKNN
from cognitiverec.models.popularity import Popularity
from cognitiverec.models.recency import RecencyProfile
from cognitiverec.models.svd import SVD

MODEL_KINDS: dict[str, type[Recommender]] = {
    "Cognitive": CognitiveRecommender,
    "EASE": EASE,
    "ItemKNN": ItemKNN,
    "Popularity": Popularity,
    "RecencyProfile": RecencyProfile,
    "SVD": SVD,
}


def build_model(kind: str, **params: Any) -> Recommender:
    """Create an unfitted model from its configuration name and hyperparameters."""
    return MODEL_KINDS[kind](**params)


__all__ = [
    "CognitiveRecommender",
    "Explainable",
    "ItemKNN",
    "MODEL_KINDS",
    "Popularity",
    "Recommender",
    "SVD",
    "build_model",
]
