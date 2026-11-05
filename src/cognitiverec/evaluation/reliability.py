"""RQ4 reliability under preference change, noise and sparse history. Owner: Mudit.

Every test here treats all models identically: same users, same injected interactions, same
targets. Each works on a copy of the model, so one test cannot affect the next.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from cognitiverec.evaluation.protocol import Phase, evaluate_ranking, rank_top_k, user_order
from cognitiverec.evaluation.quality import ndcg_at_k
from cognitiverec.models.base import Recommender

INJECTED_RATING = 5.0


def _after_history(phase: Phase) -> float:
    """Return a timestamp later than everything the model has seen."""
    return float(phase.history["timestamp"].max() + 1)


def observed_drift_users(
    phase: Phase, min_target_positives: int = 3, min_history_positives: int = 5
) -> set[int]:
    """Return users whose most-liked genre differs between history and targets.

    This is only a label for a difference between two samples. Most MovieLens users rate in a
    single sitting, so it is not evidence of a real change in taste (see docs/methodology.md).
    """
    threshold = phase.positive_threshold
    liked = phase.history[phase.history["rating"] >= threshold]
    liked_by_user = {int(user): group["item"].to_numpy() for user, group in liked.groupby("user")}
    flagged = set()
    for user, targets in phase.positives.items():
        history_items = liked_by_user.get(user)
        if history_items is None or len(history_items) < min_history_positives:
            continue
        if len(targets) < min_target_positives:
            continue
        target_items = np.fromiter(targets, dtype=np.int64)
        before = np.argmax(phase.item_genres[history_items].sum(axis=0))
        after = np.argmax(phase.item_genres[target_items].sum(axis=0))
        if before != after:
            flagged.add(user)
    return flagged


def evaluate_replay(
    model: Recommender, phase: Phase, k: int = 10, rng: np.random.Generator | None = None
) -> pd.DataFrame:
    """Feed each user's first target interactions to the model, then predict the later positives.

    All models get the same replay and are scored on the same target set. Returns one row per user
    with at least one later positive.
    """
    model = copy.deepcopy(model)
    drift = observed_drift_users(phase)
    rows = []
    for user in user_order(phase.users, rng):
        user = int(user)
        stream = phase.streams[user]
        split_at = len(stream) // 2 + 1
        replayed, later = stream.iloc[:split_at], stream.iloc[split_at:]
        targets = set(later.loc[later["rating"] >= phase.positive_threshold, "item"].tolist())
        if not targets:
            continue
        for row in replayed.itertuples(index=False):
            model.observe(user, int(row.item), float(row.rating), float(row.timestamp))
        exclude = np.concatenate([phase.seen_items(user), replayed["item"].to_numpy()])
        items = rank_top_k(model.score(user), exclude, k)
        model.record_impression(items)
        rows.append(
            {
                "user": user,
                "ndcg": ndcg_at_k(items, targets, k),
                "n_targets": len(targets),
                "observed_drift": user in drift,
            }
        )
    columns = ["user", "ndcg", "n_targets", "observed_drift"]
    return pd.DataFrame(rows, columns=columns).sort_values("user").reset_index(drop=True)


def _draw_noise(phase: Phase, user: int, n_noise: int, rng: np.random.Generator) -> np.ndarray:
    """Pick random items this user has never interacted with, in history or in targets."""
    excluded = np.zeros(phase.n_items, dtype=bool)
    excluded[phase.seen_items(user)] = True
    excluded[phase.streams[user]["item"].to_numpy()] = True
    pool = np.flatnonzero(~excluded)
    return rng.choice(pool, size=min(n_noise, len(pool)), replace=False)


def _noise_pass(
    model: Recommender, phase: Phase, order: np.ndarray, noise: dict[int, np.ndarray], k: int
) -> float:
    """Run one recommendation pass, injecting each user's noise first. Returns mean NDCG."""
    model = copy.deepcopy(model)
    timestamp = _after_history(phase)
    scores = []
    for user in order:
        user = int(user)
        injected = noise.get(user, np.empty(0, dtype=np.int64))
        for item in injected:
            model.observe(user, int(item), INJECTED_RATING, timestamp)
        exclude = np.concatenate([phase.seen_items(user), injected])
        items = rank_top_k(model.score(user), exclude, k)
        model.record_impression(items)
        scores.append(ndcg_at_k(items, phase.positives[user], k))
    return float(np.mean(scores))


def evaluate_noise(
    model: Recommender, phase: Phase, n_noise: int, k: int, rng: np.random.Generator
) -> float:
    """Return noisy NDCG as a percentage of clean NDCG.

    The two passes differ only in the noise: same model state, user order and exposure updates.
    With `n_noise=0` the result is exactly 100, and noise draws do not depend on the model, so a
    seed gives every model the same noise.
    """
    order = user_order(phase.users, rng)
    noise = {int(user): _draw_noise(phase, int(user), n_noise, rng) for user in order}
    clean = _noise_pass(model, phase, order, {}, k)
    noisy = _noise_pass(model, phase, order, noise, k)
    return 100.0 * (noisy / clean) if clean > 0 else float("nan")


def _shift_plan(
    phase: Phase, user: int, n_items: int, k: int, min_popularity: int, rng: np.random.Generator
) -> tuple[int, np.ndarray] | None:
    """Pick a genre this user rarely touches, and the items to inject from it, in order."""
    excluded = np.zeros(phase.n_items, dtype=bool)
    excluded[phase.seen_items(user)] = True
    excluded[phase.streams[user]["item"].to_numpy()] = True
    available = (phase.popularity >= min_popularity) & ~excluded

    in_genre = phase.item_genres.astype(bool)
    eligible = [
        genre
        for genre, name in enumerate(phase.genre_names)
        if name != "unknown" and (in_genre[:, genre] & available).sum() >= n_items + k
    ]
    if not eligible:
        return None
    counts = phase.item_genres[phase.seen_items(user)].sum(axis=0)
    rarest = sorted(eligible, key=lambda genre: (counts[genre], genre))[:3]
    genre = int(rng.choice(rarest))
    pool = np.flatnonzero(in_genre[:, genre] & available)
    return genre, rng.choice(pool, size=n_items, replace=False)


def evaluate_simulated_shift(
    model: Recommender,
    phase: Phase,
    steps: Iterable[int] = (1, 2, 3, 5, 10),
    k: int = 10,
    rng: np.random.Generator | None = None,
    min_item_popularity: int = 5,
) -> pd.DataFrame:
    """Inject interactions from a new genre and track adaptation against forgetting.

    Returns one row per step (including step 0):
    - `adaptation`: share of the top-k list in the new genre. How quickly did it adapt?
    - `retention_pct`: NDCG on the user's original positives, as a percentage of step 0. Did it
      keep the user's existing interests?
    """
    rng = np.random.default_rng(0) if rng is None else rng
    model = copy.deepcopy(model)
    checkpoints = sorted(set(steps))
    last_step = checkpoints[-1]
    timestamp = _after_history(phase)
    rows = []
    for user in user_order(phase.users, rng):
        user = int(user)
        plan = _shift_plan(phase, user, last_step, k, min_item_popularity, rng)
        if plan is None:
            continue
        genre, injected = plan
        in_genre = phase.item_genres[:, genre].astype(bool)
        for step in range(last_step + 1):
            if step > 0:
                model.observe(user, int(injected[step - 1]), INJECTED_RATING, timestamp)
            if step == 0 or step in checkpoints:
                exclude = np.concatenate([phase.seen_items(user), injected[:step]])
                items = rank_top_k(model.score(user), exclude, k)
                model.record_impression(items)
                rows.append(
                    {
                        "user": user,
                        "step": step,
                        "adaptation": float(in_genre[items].mean()),
                        "ndcg": ndcg_at_k(items, phase.positives[user], k),
                    }
                )
    frame = pd.DataFrame(rows, columns=["user", "step", "adaptation", "ndcg"])
    curve = (
        frame.groupby("step")
        .agg(adaptation=("adaptation", "mean"), ndcg=("ndcg", "mean"), users=("user", "nunique"))
        .reset_index()
    )
    start = curve["ndcg"].iloc[0] if len(curve) else 0.0
    curve["retention_pct"] = 100.0 * curve["ndcg"] / start if start > 0 else np.nan
    return curve


def evaluate_stability(
    model: Recommender,
    phase: Phase,
    k: int = 10,
    max_steps: int = 5,
    rng: np.random.Generator | None = None,
) -> float:
    """Return the mean top-k Jaccard overlap between consecutive lists as interactions arrive.

    An item the user just consumed cannot be recommended again, so it is removed from the earlier
    list first; otherwise every model would look unstable for a trivial reason. High stability is
    not automatically good: a model that never adapts scores 1.0.
    """
    model = copy.deepcopy(model)
    overlaps = []
    for user in user_order(phase.users, rng):
        user = int(user)
        exclude = phase.seen_items(user)
        previous = rank_top_k(model.score(user), exclude, k)
        model.record_impression(previous)
        for row in phase.streams[user].iloc[:max_steps].itertuples(index=False):
            item = int(row.item)
            model.observe(user, item, float(row.rating), float(row.timestamp))
            exclude = np.append(exclude, item)
            current = rank_top_k(model.score(user), exclude, k)
            model.record_impression(current)
            before = {int(value) for value in previous} - {item}
            after = {int(value) for value in current}
            union = before | after
            overlaps.append(len(before & after) / len(union) if union else 1.0)
            previous = current
    return float(np.mean(overlaps)) if overlaps else float("nan")


def evaluate_cold_start(
    factory: Callable[[], Recommender],
    phase: Phase,
    sizes: Iterable[int] = (5, 10, 20),
    fraction: float = 0.2,
    k: int = 10,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Refit with a random group of users cut down to their first n interactions.

    Everyone else keeps their full history, so this tests sparse users rather than a sparse system.
    All users are served through the normal protocol, in one fixed order, and NDCG is reported for
    the cut users. Candidates still exclude their full history, which keeps results comparable with
    `cold_ndcg_full`: the same users' NDCG with nothing removed.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n_cold = max(1, int(fraction * len(phase.users)))
    cold_users = rng.choice(phase.users, size=n_cold, replace=False)
    is_cold = phase.history["user"].isin(cold_users)
    order_seed = int(rng.integers(2**32))

    def mean_ndcg(model: Recommender) -> float:
        per_user = evaluate_ranking(model, phase, k, np.random.default_rng(order_seed)).per_user
        return float(per_user.loc[per_user["user"].isin(cold_users), "ndcg"].mean())

    results = {"cold_ndcg_full": mean_ndcg(phase.fit(factory()))}
    for size in sizes:
        cut = phase.history[is_cold].groupby("user", sort=False).head(size)
        truncated = pd.concat([phase.history[~is_cold], cut]).sort_values("user", kind="stable")
        model = factory().fit(truncated, phase.n_users, phase.n_items)
        results[f"cold_ndcg_{size}"] = mean_ndcg(model)
    return results
