"""Bootstrap confidence intervals for per-user comparisons. Shared."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Return the mean and a percentile bootstrap interval, ignoring NaNs."""
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[~np.isnan(clean)]
    if len(clean) == 0:
        return (np.nan, np.nan, np.nan)
    resampled = rng.choice(clean, size=(n_boot, len(clean)), replace=True).mean(axis=1)
    low, high = np.percentile(resampled, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(clean.mean()), float(low), float(high))
