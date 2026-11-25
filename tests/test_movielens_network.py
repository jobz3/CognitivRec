"""Checks against the real MovieLens 100K data. Run with `make test-network`."""

import numpy as np
import pytest

from cognitiverec.config import load_config, load_tuned, resolve_path
from cognitiverec.data import chronological_split, load_movielens_100k, temporal_report
from cognitiverec.evaluation.protocol import (
    Phase,
    evaluate_ranking,
    test_phase,
    validation_phase,
)
from cognitiverec.evaluation.reliability import evaluate_noise
from cognitiverec.experiments.stage2_cognitive import memory_report
from cognitiverec.models import SVD, CognitiveRecommender, ItemKNN, Popularity

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def data():
    config = load_config()
    dataset = load_movielens_100k(resolve_path(config, "data"))
    split = chronological_split(dataset.interactions, 0.70, 0.15, mode="per_user")
    return dataset, split


def test_dataset_and_split_sizes(data):
    dataset, split = data
    assert (dataset.n_users, dataset.n_items, len(dataset.interactions)) == (943, 1682, 100_000)
    assert (len(split.train), len(split.val), len(split.test)) == (69_572, 14_973, 15_455)
    assert dataset.item_titles[0] == "Toy Story (1995)"


def test_temporal_report_matches_the_documented_limits(data):
    report = temporal_report(data[1])
    assert report["single_session_users_pct"] == pytest.approx(59.6, abs=0.1)
    assert report["same_second_ratings_pct"] == pytest.approx(75.8, abs=0.1)
    assert report["train_later_than_test_row_pct"] == pytest.approx(39.9, abs=0.1)


def test_global_split_uses_only_prior_interactions_and_a_warm_catalogue(data):
    dataset, _ = data
    split = chronological_split(dataset.interactions, mode="global")
    assert len(split.train) + len(split.val) + len(split.test) == 100_000
    assert split.train["timestamp"].max() < split.val["timestamp"].min()
    assert split.val["timestamp"].max() < split.test["timestamp"].min()
    phase = test_phase(dataset, split)
    assert temporal_report(split)["train_later_than_test_row_pct"] == 0
    assert 0 < phase.cohort["cold_user_rows"] < len(split.test)
    assert 0 < phase.cohort["unavailable_item_rows"] < len(split.test)
    assert set(phase.targets["user"]) <= set(phase.history["user"])
    assert set(phase.targets["item"]) <= set(phase.history["item"])
    result = evaluate_ranking(phase.fit(Popularity()), phase)
    for user, items in result.recommendations.items():
        assert phase.catalogue[items].all()
        assert not np.intersect1d(items, phase.seen_items(user)).size


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (Popularity, 0.0882),
        (lambda: ItemKNN(top_k=80, shrinkage=15.0), 0.1332),
        (lambda: SVD(n_factors=32), 0.0839),
    ],
    ids=["popularity", "item_knn", "svd"],
)
def test_baselines_reproduce_the_plans_numbers_once_validation_is_masked(data, build, expected):
    """Fit on train only with the plan's parameters: the port must not change the baselines."""
    dataset, split = data
    masked = Phase("regression", test_phase(dataset, split).history, split.test, dataset)
    model = build().fit(split.train, dataset.n_users, dataset.n_items)
    assert evaluate_ranking(model, masked).per_user["ndcg"].mean() == pytest.approx(
        expected, abs=0.002
    )


def test_short_term_memory_reaches_the_ranking(data):
    """As written, the plan's code measures 0.045 and 99.8% on this test."""
    dataset, split = data
    phase = validation_phase(dataset, split)
    for params in ({}, load_tuned(load_config(), "cognitive")["model_c"]):
        report = memory_report(phase.fit(CognitiveRecommender(**params)), phase)
        assert report["short_to_long_term_spread"] >= 0.3
        assert report["attention_on_newest_memory_pct"] <= 90.0


def test_zero_noise_retention_is_exactly_100_on_real_data(data):
    dataset, split = data
    phase = validation_phase(dataset, split)
    tuned = load_tuned(load_config(), "cognitive")["model_d"]
    model = phase.fit(CognitiveRecommender(**tuned))
    assert evaluate_noise(model, phase, 0, 10, np.random.default_rng(1)) == 100.0
