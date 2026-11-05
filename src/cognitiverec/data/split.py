"""Per-user chronological split and a report of what it does not protect against. Owner: Mudit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SESSION_GAP_SECONDS = 1800


@dataclass(frozen=True)
class Split:
    """Interaction frames, each ordered by user and then by time."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    interactions: pd.DataFrame, train: float = 0.70, val: float = 0.15
) -> Split:
    """Give each user the first `train` of their ratings, then `val`, then the rest.

    This stops a user's own future reaching their training data, but other users' later ratings
    can still appear in training. `temporal_report` measures that, and docs/methodology.md
    explains why this split is still used.

    Ratings that share a timestamp stay in dataset order, because the sort is stable.
    """
    ordered = interactions.sort_values(["user", "timestamp"], kind="stable").reset_index(drop=True)
    val_end = round(train + val, 10)  # 0.70 + 0.15 is not exactly 0.85 in floating point
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
