import numpy as np
import pandas as pd
import pytest

from cognitiverec.evaluation.fairness import (
    activity_gap,
    band_report,
    catalogue_coverage,
    exposure_counts,
    gini,
    popularity_bands,
)
from cognitiverec.evaluation.protocol import RankingRun


def test_gini_extremes():
    assert gini(np.array([5, 5, 5, 5])) == pytest.approx(0.0)
    assert gini(np.array([0, 0, 0, 1])) == pytest.approx(0.75)
    assert gini(np.zeros(3)) == 0.0


def test_exposure_counts_and_coverage():
    recommendations = {0: np.array([0, 1]), 1: np.array([1, 2])}
    exposure = exposure_counts(recommendations, 5)
    np.testing.assert_array_equal(exposure, [1, 2, 1, 0, 0])
    assert catalogue_coverage(exposure) == pytest.approx(60.0)


def test_bands_hold_about_a_third_of_interactions_each():
    popularity = np.array([30, 30, 10, 10, 10, 5, 5])
    # Cumulative share by item: 0.30, 0.60, 0.70, ...
    np.testing.assert_array_equal(popularity_bands(popularity), [0, 0, 1, 2, 2, 2, 2])


def test_head_band_is_never_empty_when_one_item_dominates():
    # Item 0 alone passes both boundaries, so nothing is left for the mid band.
    np.testing.assert_array_equal(popularity_bands(np.array([90, 5, 5])), [0, 2, 2])


def test_merit_gap_is_zero_when_exposure_follows_relevance():
    bands = np.array([0, 1, 2])
    report = band_report({0: np.array([0, 1, 2])}, {0: {0, 1, 2}}, bands)
    assert report["merit_gap"] == pytest.approx(0.0)
    assert report["recall_tail"] == pytest.approx(1.0)


def test_merit_gap_is_large_when_only_the_head_is_shown():
    """Showing only popular items when relevance sits in the tail is not fair, however it looks."""
    bands = np.array([0, 0, 2, 2])
    report = band_report({0: np.array([0, 1])}, {0: {2, 3}}, bands)
    assert report["merit_gap"] == pytest.approx(1.0)
    assert report["recall_tail"] == pytest.approx(0.0)
    assert np.isnan(report["recall_head"])


def test_activity_gap_compares_low_and_high_activity_users():
    class FakePhase:
        seen = {0: np.arange(1), 1: np.arange(2), 2: np.arange(30), 3: np.arange(40)}

        def seen_items(self, user):
            return self.seen[user]

    per_user = pd.DataFrame({"user": [0, 1, 2, 3], "ndcg": [0.1, 0.1, 0.5, 0.5]})
    assert activity_gap(RankingRun(per_user, {}), FakePhase()) == pytest.approx(0.4)
