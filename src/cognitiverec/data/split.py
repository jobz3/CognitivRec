"""Chronological splits and temporal diagnostics. Owner: Mudit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SESSION_GAP_SECONDS = 1800


@dataclass(frozen=True)
class Split:
    """Interaction frames and the protocol used to partition them."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    mode: str = "per_user"


def chronological_split(
    interactions: pd.DataFrame,
    train: float = 0.70,
    val: float = 0.15,
    mode: str = "per_user",
    test_end: float = 1.0,
) -> Split:
    """Split by rating timestamps, with a global cutoff or a legacy per-user split.

    Global cutoffs move to the start of a timestamp group, so no timestamp crosses a boundary.
    The legacy protocol takes the requested proportions independently for each user.
    """
    val_end = round(train + val, 10)
    if train <= 0 or val <= 0 or val_end > 1:
        raise ValueError("Invalid train and validation fractions.")
    if mode == "global":
        if not val_end < test_end <= 1:
            raise ValueError("A global split requires a nonempty test window.")
        ordered = interactions.sort_values("timestamp", kind="stable").reset_index(drop=True)
        times = ordered["timestamp"].to_numpy()
        if not len(times):
            raise ValueError("A global split requires timestamped interactions.")
        cutoffs = times[[int(train * len(times)), int(val_end * len(times))]]
        first, second = np.searchsorted(times, cutoffs)
        end = len(times) if test_end == 1 else np.searchsorted(times, times[int(test_end * len(times))])
        if not 0 < first < second < end:
            raise ValueError("Global cutoffs require three nonempty, distinct timestamp windows.")
        return Split(
            ordered.iloc[:first].reset_index(drop=True),
            ordered.iloc[first:second].reset_index(drop=True),
            ordered.iloc[second:end].reset_index(drop=True),
            mode=mode,
        )
    if mode != "per_user":
        raise ValueError(f"Unknown temporal split mode: {mode}")
    if test_end != 1:
        raise ValueError("A capped test window requires the global split.")
    ordered = interactions.sort_values(["user", "timestamp"], kind="stable").reset_index(drop=True)
    train_parts, val_parts, test_parts = [], [], []
    for _, group in ordered.groupby("user", sort=False):
        n = len(group)
        first = max(1, int(train * n))
        second = max(1, int(val_end * n))
        train_parts.append(group.iloc[:first])
        val_parts.append(group.iloc[first:second])
        test_parts.append(group.iloc[second:])
    return Split(
        train=pd.concat(train_parts).reset_index(drop=True),
        val=pd.concat(val_parts).reset_index(drop=True),
        test=pd.concat(test_parts).reset_index(drop=True),
    )


def temporal_report(split: Split) -> dict[str, float]:
    """Measure how much real time, and how much global ordering, the split actually has."""
    everything = pd.concat([split.train, split.val, split.test]).sort_values(
        ["user", "timestamp"], kind="stable"
    )
    by_user = everything.groupby("user")["timestamp"]
    span = by_user.max() - by_user.min()
    sessions = (by_user.diff() > SESSION_GAP_SECONDS).groupby(everything["user"]).sum() + 1
    gap = (
        split.test.groupby("user")["timestamp"].min()
        - split.train.groupby("user")["timestamp"].max()
    )
    train_times = np.sort(split.train["timestamp"].to_numpy())
    later = 1 - np.searchsorted(train_times, split.test["timestamp"].to_numpy(), side="right") / (
        len(train_times)
    )
    return {
        "median_user_span_hours": float(span.median() / 3600),
        "single_session_users_pct": float(100 * (sessions == 1).mean()),
        "same_second_ratings_pct": float(
            100 * everything.duplicated(["user", "timestamp"], keep=False).mean()
        ),
        "median_train_to_test_gap_minutes": float(gap.median() / 60),
        "train_later_than_test_row_pct": float(100 * later.mean()),
    }
