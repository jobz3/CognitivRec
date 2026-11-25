"""RQ3 explanation fidelity through leave-one-out deletion. Owner: Abhinav.

For each user's top recommendation, every piece of evidence is deleted in turn. A faithful
explanation names evidence whose removal matters more than the rest. Following Jain and Wallace
(2019), attributions are also rank-correlated with the actual deletion effect.

Comparing against the exact mean over all other entries, rather than one random entry, removes
deletion sampling noise. Models with no attributions report NaN.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from cognitiverec.evaluation.protocol import Phase, rank_top_k, user_order
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


def _user_fidelity(
    model: Explainable, user: int, candidates: np.ndarray, base: np.ndarray, top: int
) -> dict | None:
    """Delete each evidence entry for one user's top item and measure the effect."""
    weights = model.attributions(user, top)
    if len(weights) < MIN_ENTRIES:
        return None

    drops = np.empty(len(weights))
    counterfactual_tops = np.empty(len(weights), dtype=np.int64)
    for entry in range(len(weights)):
        counterfactual = model.score_without(user, entry)
        drops[entry] = base[top] - counterfactual[top]
        counterfactual_tops[entry] = np.where(candidates, counterfactual, -np.inf).argmax()

    spread = float(base[candidates].std())
    if spread == 0.0:
        return None
    strongest = int(np.argmax(weights))
    others = np.delete(np.arange(len(weights)), strongest)
    top_changed = counterfactual_tops != top
    return {
        "gap": (drops[strongest] - drops[others].mean()) / spread,
        "strongest_drop": drops[strongest] / spread,
        "other_drop": drops[others].mean() / spread,
        "score_rises": drops[strongest] < 0,
        "top1_change_strongest": top_changed[strongest],
        "top1_change_other": top_changed[others].mean(),
        "tau": kendalltau(weights, drops).statistic,
        "evidence_item": int(model.evidence_items(user)[strongest]),
        "evidence_weight": float(weights[strongest]),
        "base_score": float(base[top]),
        "score_after_deletion": float(base[top] - drops[strongest]),
        "top_after_deletion": int(counterfactual_tops[strongest]),
    }


def evaluate_fidelity(
    model: Recommender,
    phase: Phase,
    k: int = 10,
    rng: np.random.Generator | None = None,
    examples: list[dict] | None = None,
    *,
    demo_cases: bool = False,
) -> dict[str, float]:
    """Measure whether a model's attributions identify the evidence that drives its top pick.

    `fidelity_gap` is the mean per-user difference between deleting the strongest entry and
    deleting any other, in units of that user's candidate score spread. Positive values mean
    attributions point at evidence that matters. Deletion effects are averaged per user and then
    across users, never turned into a ratio, because the denominator of a ratio can be near zero.

    Deletions hold exposure fixed at the actual decision state. The selected recommendation list
    then updates exposure, following the same order and impressions as the corresponding ranking.
    """
    results = dict.fromkeys(FIDELITY_KEYS, np.nan)
    results["fidelity_users"] = 0
    if not isinstance(model, Explainable):
        return results
    model = copy.deepcopy(model)

    rows = []
    for user in user_order(phase.users, rng):
        user = int(user)
        exclude = phase.excluded_items(user)
        scores = model.score(user)
        items = rank_top_k(scores, exclude, k)
        if not len(items):
            continue
        candidates = np.ones(phase.n_items, dtype=bool)
        candidates[exclude] = False
        top = int(items[0])
        row = _user_fidelity(model, user, candidates, scores, top)
        if row is not None:
            row.update(user=user, top_item=top)
            if hasattr(model, "score_components"):
                row.update(model.score_components(user, top))
            rows.append(row)
        model.record_impression(items)
    if not rows:
        return results

    frame = pd.DataFrame(rows)
    if examples is not None:
        ordered = frame.sort_values(["gap", "user"])
        if demo_cases:
            selections = {
                "top recommendation changes": ordered[ordered["top1_change_strongest"]].tail(1),
                "original score rises": ordered[ordered["score_rises"]].head(1),
            }
        else:
            selections = {
                label: ordered.iloc[[position]]
                for label, position in (
                    ("lowest", 0),
                    ("median", len(ordered) // 2),
                    ("highest", -1),
                )
            }
        for label, selection in selections.items():
            if selection.empty:
                continue
            example = selection.iloc[0].to_dict()
            examples.append(
                {
                    **example,
                    "selection": label,
                    "recommended_title": phase.item_titles[int(example["top_item"])],
                    "evidence_title": phase.item_titles[int(example["evidence_item"])],
                    "top_after_deletion_title": phase.item_titles[
                        int(example["top_after_deletion"])
                    ],
                }
            )
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
