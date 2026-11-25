"""RQ2 fairness metrics. Owner: Jobin.

Exposure equality on its own is not fairness, because items differ in relevance. So exposure
inequality (coverage, Gini) is reported next to relevance-aware measures. PROJECT_REPORT.md
states the definitions used.
"""

from __future__ import annotations

import numpy as np

from cognitiverec.evaluation.protocol import Phase, RankingRun

BAND_NAMES = ("head", "mid", "tail")


def exposure_counts(recommendations: dict[int, np.ndarray], n_items: int) -> np.ndarray:
    """Count how many recommendation lists each item appears in."""
    exposure = np.zeros(n_items, dtype=np.float64)
    for items in recommendations.values():
        exposure[items] += 1.0
    return exposure


def catalogue_coverage(exposure: np.ndarray) -> float:
    """Return the percentage of the catalogue recommended to at least one user."""
    return float(100.0 * np.mean(exposure > 0))


def gini(exposure: np.ndarray) -> float:
    """Return the Gini index of item exposure: 0 is perfect equality, near 1 is a monopoly."""
    values = np.sort(np.asarray(exposure, dtype=np.float64))
    total = values.sum()
    if total == 0:
        return 0.0
    n = len(values)
    return float(np.sum((2 * np.arange(1, n + 1) - n - 1) * values) / (n * total))


def popularity_bands(popularity: np.ndarray) -> np.ndarray:
    """Label items 0 (head), 1 (mid) or 2 (tail), so each band holds about a third of interactions.

    A band includes the item that crosses its boundary, so head is never empty even if one item
    alone holds more than a third of all interactions.
    """
    order = np.argsort(-popularity, kind="stable")
    share = np.cumsum(popularity[order]) / max(popularity.sum(), 1)
    head_end = int(np.searchsorted(share, 1 / 3)) + 1
    mid_end = int(np.searchsorted(share, 2 / 3)) + 1
    bands = np.full(len(popularity), 2, dtype=np.int64)
    bands[order[head_end:mid_end]] = 1
    bands[order[:head_end]] = 0
    return bands


def band_report(
    recommendations: dict[int, np.ndarray], positives: dict[int, set[int]], bands: np.ndarray
) -> dict[str, float]:
    """Compare each popularity band's share of exposure with its share of relevant items.

    `merit_gap` is half the summed absolute difference: 0 means exposure is exactly proportional
    to observed relevance. `recall_<band>` is the chance that a relevant item in that band gets
    recommended, which should not depend on popularity under equal opportunity.
    """
    exposed = np.zeros(3)
    rank_exposed = np.zeros(3)
    relevant = np.zeros(3)
    hit = np.zeros(3)
    for user, items in recommendations.items():
        np.add.at(exposed, bands[items], 1.0)
        np.add.at(rank_exposed, bands[items], 1.0 / np.log2(np.arange(len(items)) + 2))
        targets = positives[user]
        np.add.at(relevant, bands[np.fromiter(targets, dtype=np.int64)], 1.0)
        recommended = [int(item) for item in items if int(item) in targets]
        np.add.at(hit, bands[np.array(recommended, dtype=np.int64)], 1.0)

    exposure_share = exposed / max(exposed.sum(), 1.0)
    rank_exposure_share = rank_exposed / max(rank_exposed.sum(), 1.0)
    relevance_share = relevant / max(relevant.sum(), 1.0)
    report: dict[str, float] = {
        "merit_gap": float(0.5 * np.abs(exposure_share - relevance_share).sum()),
        "rank_merit_gap": float(0.5 * np.abs(rank_exposure_share - relevance_share).sum()),
    }
    for index, band in enumerate(BAND_NAMES):
        report[f"exposure_share_{band}"] = float(exposure_share[index])
        report[f"rank_exposure_share_{band}"] = float(rank_exposure_share[index])
        report[f"relevance_share_{band}"] = float(relevance_share[index])
        report[f"relevant_{band}"] = int(relevant[index])
        report[f"recall_{band}"] = (
            float(hit[index] / relevant[index]) if relevant[index] else np.nan
        )
    return report


def activity_report(run: RankingRun, phase: Phase) -> dict[str, float]:
    """Report group sizes, NDCG and its absolute gap for the two activity groups.

    Groups split at the median history size. The dataset's demographic fields are not used, so no
    demographic fairness claim follows from this.
    """
    sizes = run.per_user["user"].map(lambda user: len(phase.seen_items(int(user))))
    low = sizes <= sizes.median()
    low_mean = float(run.per_user.loc[low, "ndcg"].mean())
    high_mean = float(run.per_user.loc[~low, "ndcg"].mean())
    return {
        "activity_gap": abs(high_mean - low_mean),
        "low_activity_ndcg": low_mean,
        "high_activity_ndcg": high_mean,
        "low_activity_users": int(low.sum()),
        "high_activity_users": int((~low).sum()),
    }


def activity_gap(run: RankingRun, phase: Phase) -> float:
    return activity_report(run, phase)["activity_gap"]


def fairness_report(run: RankingRun, phase: Phase) -> dict[str, float]:
    """All RQ2 metrics for one ranking run."""
    exposure = exposure_counts(run.recommendations, phase.n_items)
    report = {
        "coverage_pct": catalogue_coverage(exposure[phase.catalogue]),
        "gini": gini(exposure[phase.catalogue]),
        **activity_report(run, phase),
    }
    report.update(
        band_report(run.recommendations, phase.positives, popularity_bands(phase.popularity))
    )
    return report
