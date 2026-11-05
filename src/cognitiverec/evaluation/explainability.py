"""RQ3 explanation fidelity through leave-one-out deletion. Owner: Abhinav.

For each user's top recommendation, every piece of evidence is deleted in turn. A faithful
explanation names evidence whose removal matters more than the rest. Following Jain and Wallace
(2019), attributions are also rank-correlated with the actual deletion effect.

Comparing against the exact mean over all other entries, rather than one random entry, removes
sampling noise, so these results do not depend on a seed. Models with no attributions report NaN:
they were not tested, which is different from scoring zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from cognitiverec.evaluation.protocol import Phase
from cognitiverec.evaluation.stats import bootstrap_ci
from cognitiverec.models.base import Explainable, Recommender

MIN_ENTRIES = 3
FIDELITY_KEYS = (
    "fidelity_gap",
    "fidelity_gap_low",
    "fidelity_gap_high",
    "strongest_drop",
    "other_drop",
    "score_rises_pct",
    "top1_change_strongest_pct",
    "top1_change_other_pct",
    "kendall_tau",
    "fidelity_users",
)


def _user_fidelity(model: Explainable, user: int, candidates: np.ndarray) -> dict | None:
    """Delete each evidence entry for one user's top item and measure the effect."""
    base = model.score(user)
    top = int(np.where(candidates, base, -np.inf).argmax())
    weights = model.attributions(user, top)
    if len(weights) < MIN_ENTRIES:
        return None

    drops = np.empty(len(weights))
    top_changed = np.empty(len(weights), dtype=bool)
    for entry in range(len(weights)):
        counterfactual = model.score_without(user, entry)
        drops[entry] = base[top] - counterfactual[top]
        top_changed[entry] = int(np.where(candidates, counterfactual, -np.inf).argmax()) != top

    spread = float(base[candidates].std())
    if spread == 0.0:
        return None
    strongest = int(np.argmax(weights))
    others = np.delete(np.arange(len(weights)), strongest)
    return {
        "gap": (drops[strongest] - drops[others].mean()) / spread,
        "strongest_drop": drops[strongest] / spread,
        "other_drop": drops[others].mean() / spread,
        "score_rises": drops[strongest] < 0,
        "top1_change_strongest": top_changed[strongest],
        "top1_change_other": top_changed[others].mean(),
        "tau": kendalltau(weights, drops).statistic,
    }


def evaluate_fidelity(model: Recommender, phase: Phase) -> dict[str, float]:
    """Measure whether a model's attributions identify the evidence that drives its top pick.

    `fidelity_gap` is the mean per-user difference between deleting the strongest entry and
    deleting any other, in units of that user's candidate score spread. Positive values mean
    attributions point at evidence that matters. Deletion effects are averaged per user and then
    across users, never turned into a ratio, because the denominator of a ratio can be near zero.

    No impressions are recorded, so Model D's exposure stays as fitted throughout this test and the
    result does not depend on user order.
    """
    results = dict.fromkeys(FIDELITY_KEYS, np.nan)
    results["fidelity_users"] = 0
    if not isinstance(model, Explainable):
        return results

    rows = []
    for user in phase.users:
        user = int(user)
        candidates = np.ones(phase.n_items, dtype=bool)
        candidates[phase.seen_items(user)] = False
        if not candidates.any():
            continue
        row = _user_fidelity(model, user, candidates)
        if row is not None:
            rows.append(row)
    if not rows:
        return results

    frame = pd.DataFrame(rows)
    # A fixed generator: the interval should not vary between otherwise identical runs.
    mean, low, high = bootstrap_ci(frame["gap"].to_numpy(), np.random.default_rng(0))
    results.update(
        fidelity_gap=mean,
        fidelity_gap_low=low,
        fidelity_gap_high=high,
        strongest_drop=float(frame["strongest_drop"].mean()),
        other_drop=float(frame["other_drop"].mean()),
        score_rises_pct=float(100 * frame["score_rises"].mean()),
        top1_change_strongest_pct=float(100 * frame["top1_change_strongest"].mean()),
        top1_change_other_pct=float(100 * frame["top1_change_other"].mean()),
        kendall_tau=float(frame["tau"].mean()),
        fidelity_users=len(frame),
    )
    return results
