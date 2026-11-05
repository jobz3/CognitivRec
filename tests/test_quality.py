import math

import numpy as np
import pytest

from cognitiverec.evaluation.quality import (
    genre_diversity,
    ndcg_at_k,
    ranking_metrics,
    self_information,
)


def test_ndcg_matches_a_hand_calculation():
    # Hits at ranks 1 and 3, with two relevant items in total.
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(np.array([1, 2, 3]), {1, 3}, 3) == pytest.approx(dcg / ideal)


def test_ndcg_extremes():
    assert ndcg_at_k(np.array([1, 2]), {1, 2}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(np.array([3, 4]), {1, 2}, 2) == 0.0
    assert ndcg_at_k(np.array([3, 4]), set(), 2) == 0.0


def test_ndcg_ideal_is_capped_at_k():
    """With more relevant items than slots, a full list of hits is still a perfect score."""
    assert ndcg_at_k(np.array([1, 2]), {1, 2, 3, 4, 5}, 2) == pytest.approx(1.0)


def test_precision_and_recall():
    metrics = ranking_metrics(
        np.array([1, 2, 3, 4]), {1, 3, 9}, 4, np.zeros(10), np.eye(10, dtype=np.int8)
    )
    assert metrics["precision"] == pytest.approx(2 / 4)
    assert metrics["recall"] == pytest.approx(2 / 3)


def test_self_information_is_higher_for_rarer_items():
    information = self_information(np.array([0, 9]), 9)
    assert information[0] == pytest.approx(-math.log2(1 / 10))
    assert information[1] == pytest.approx(0.0)


def test_genre_diversity_extremes():
    same = np.array([[1, 0], [1, 0]], dtype=np.int8)
    different = np.array([[1, 0], [0, 1]], dtype=np.int8)
    assert genre_diversity(np.array([0, 1]), same) == pytest.approx(0.0)
    assert genre_diversity(np.array([0, 1]), different) == pytest.approx(1.0)
    assert genre_diversity(np.array([0]), same) == 0.0
