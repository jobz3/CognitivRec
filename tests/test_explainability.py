import math

import numpy as np
import pytest

from cognitiverec.evaluation.explainability import FIDELITY_KEYS, evaluate_fidelity
from cognitiverec.models import SVD, CognitiveRecommender, ItemKNN, Popularity

MEASURED_KEYS = [key for key in FIDELITY_KEYS if key != "fidelity_users"]


@pytest.mark.parametrize("build", [Popularity, lambda: SVD(n_factors=4)])
def test_models_without_attributions_report_nan_not_zero(build, val_phase):
    results = evaluate_fidelity(val_phase.fit(build()), val_phase)
    assert results["fidelity_users"] == 0
    assert all(math.isnan(results[key]) for key in MEASURED_KEYS)


def test_cognitive_model_without_memory_has_nothing_to_attribute(val_phase):
    model = val_phase.fit(CognitiveRecommender(n_factors=4, use_short_term=False))
    assert math.isnan(evaluate_fidelity(model, val_phase)["fidelity_gap"])


def test_fidelity_needs_no_seed_and_is_identical_between_runs(val_phase):
    model = val_phase.fit(CognitiveRecommender(n_factors=4, memory_size=6))
    assert evaluate_fidelity(model, val_phase) == evaluate_fidelity(model, val_phase)


def test_item_knn_attributions_are_faithful_by_construction(val_phase):
    """Each attribution equals its deletion effect exactly, so rank correlation must be 1."""
    results = evaluate_fidelity(val_phase.fit(ItemKNN(top_k=10, shrinkage=1.0)), val_phase)
    assert results["fidelity_users"] > 0
    assert results["kendall_tau"] == pytest.approx(1.0)
    assert results["fidelity_gap"] > 0
    assert results["score_rises_pct"] == 0.0


def test_fidelity_reports_an_interval_around_the_gap(val_phase):
    model = val_phase.fit(CognitiveRecommender(n_factors=4, memory_size=6))
    results = evaluate_fidelity(model, val_phase)
    assert results["fidelity_users"] > 0
    assert results["fidelity_gap_low"] <= results["fidelity_gap"] <= results["fidelity_gap_high"]


def test_gap_is_strongest_drop_minus_other_drop(val_phase):
    results = evaluate_fidelity(val_phase.fit(CognitiveRecommender(n_factors=4)), val_phase)
    assert results["fidelity_gap"] == pytest.approx(
        results["strongest_drop"] - results["other_drop"]
    )


def test_cognitive_attributions_sum_to_one_and_deletion_leaves_memory_alone(val_phase):
    model = val_phase.fit(CognitiveRecommender(n_factors=4, memory_size=6))
    before = list(model.memory[0])
    assert model.attributions(0, 5).sum() == pytest.approx(1.0)
    without = model.score_without(0, 0)
    assert model.memory[0] == before
    assert not np.array_equal(without, model.score(0))


def test_cognitive_deletion_matches_a_model_that_never_had_the_entry(val_phase):
    model = val_phase.fit(CognitiveRecommender(n_factors=4, memory_size=6))
    without = model.score_without(0, 1)
    model.memory[0] = model.memory[0][:1] + model.memory[0][2:]
    np.testing.assert_array_equal(without, model.score(0))
