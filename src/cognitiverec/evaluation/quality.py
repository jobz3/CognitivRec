"""RQ1 recommendation-quality metrics. Shared."""

from __future__ import annotations

import numpy as np


def ndcg_at_k(items: np.ndarray, positives: set[int], k: int) -> float:
    """Binary-relevance NDCG, normalised by the best ordering achievable for this user."""
    gains = [
        1.0 / np.log2(rank + 2) for rank, item in enumerate(items[:k]) if int(item) in positives
    ]
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(k, len(positives))))
    return float(sum(gains) / ideal) if ideal > 0 else 0.0


def self_information(popularity: np.ndarray, n_users: int) -> np.ndarray:
    """Item novelty in bits: rarely seen items carry more information."""
    return -np.log2((popularity + 1.0) / (n_users + 1.0))


def genre_diversity(items: np.ndarray, item_genres: np.ndarray) -> float:
    """Return 1 minus the mean pairwise genre Jaccard similarity within the list."""
    if len(items) < 2:
        return 0.0
    genres = item_genres[items].astype(bool)
    intersection = (genres[:, None, :] & genres[None, :, :]).sum(axis=2)
    union = (genres[:, None, :] | genres[None, :, :]).sum(axis=2)
    similarity = np.divide(
        intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0
    )
    upper = np.triu_indices(len(items), k=1)
    return float(1.0 - similarity[upper].mean())


def ranking_metrics(
    items: np.ndarray,
    positives: set[int],
    k: int,
    item_self_information: np.ndarray,
    item_genres: np.ndarray,
) -> dict[str, float]:
    """All RQ1 metrics for one recommendation list."""
    hits = sum(int(item) in positives for item in items[:k])
    return {
        "precision": hits / k,
        "recall": hits / len(positives) if positives else 0.0,
        "ndcg": ndcg_at_k(items, positives, k),
        "novelty": float(item_self_information[items].mean()) if len(items) else 0.0,
        "diversity": genre_diversity(items, item_genres),
    }
