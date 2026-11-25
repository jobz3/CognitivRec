"""Rating weights and normalisation shared by the latent-profile models."""

from __future__ import annotations

import numpy as np


def rating_to_reward(rating: np.ndarray | float) -> np.ndarray | float:
    """Express 1–5 star ratings on a signed scale with a neutral point of 2.5."""
    return np.clip((rating - 2.5) / 2.5, -1.0, 1.0)


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Scale each nonzero row to unit length."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 1e-10)


def unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-10 else np.zeros_like(vector)
