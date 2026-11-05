import math

import numpy as np
import pytest

from cognitiverec.models import SVD, CognitiveRecommender
from cognitiverec.models.cognitive import _unit_rows
from cognitiverec.models.exploration import ExposureTracker, ucb_bonus

EMBEDDINGS = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])


def tiny_model(**params) -> CognitiveRecommender:
    """A model with hand-set state, so expected values can be worked out exactly."""
    defaults = {"base_alpha": 0.5, "decay_lambda": 0.1, "attention_temperature": 1.0}
    model = CognitiveRecommender(**{**defaults, **params})
    model._unit = EMBEDDINGS
    model._set_embeddings(EMBEDDINGS, n_users=1)
    model.memory[0] = [0, 1]
    model.u_lt[0] = [1.0, 0.0]
    model.exposure = ExposureTracker(np.zeros(3))
    return model


def expected_reward_for_item_2() -> float:
    """The plan's formula, worked by hand for candidate item 2 with memory [item 0, item 1]."""
    similarity_0, similarity_1 = 0.6, 0.8  # v_j . v_2
    logit_0 = similarity_0 / 1.0 - 0.1 * 1  # older memory, lag 1
    logit_1 = similarity_1 / 1.0 - 0.1 * 0  # newest memory, lag 0
    total = math.exp(logit_0) + math.exp(logit_1)
    weight_0, weight_1 = math.exp(logit_0) / total, math.exp(logit_1) / total
    short_term = weight_0 * similarity_0 + weight_1 * similarity_1  # u_ST(2) . v_2
    long_term = 0.6  # u_LT . v_2
    return 0.5 * long_term + 0.5 * short_term


def test_reward_matches_the_formula_worked_by_hand():
    """Guards against the plan code's stray division by sqrt(d) in the short-term score."""
    assert tiny_model().score(0)[2] == pytest.approx(expected_reward_for_item_2())


def test_attention_weights_are_a_distribution_for_each_candidate():
    weights, _ = tiny_model()._short_term([0, 1])
    np.testing.assert_allclose(weights.sum(axis=0), 1.0)


def test_attention_depends_on_the_candidate():
    weights, _ = tiny_model()._short_term([0, 1])
    assert weights[0, 0] > weights[0, 1]  # item 0 attends to its own memory more than item 1 does


def test_lag_favours_recent_memories_when_similarity_is_equal():
    model = tiny_model(decay_lambda=0.5)
    model._unit = model._candidates = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    weights, _ = model._short_term([0, 1])
    assert weights[1, 2] > weights[0, 2]


def test_lower_temperature_sharpens_attention():
    sharp = tiny_model(attention_temperature=0.1, decay_lambda=0.0)._short_term([0, 1])[0]
    soft = tiny_model(attention_temperature=5.0, decay_lambda=0.0)._short_term([0, 1])[0]
    assert sharp[0, 0] > soft[0, 0]


def test_uniform_attention_weights_every_memory_equally():
    weights, _ = tiny_model(attention_mode="uniform")._short_term([0, 1])
    np.testing.assert_allclose(weights, 0.5)


def test_unknown_attention_mode_is_rejected():
    with pytest.raises(ValueError, match="attention_mode"):
        CognitiveRecommender(attention_mode="other")


def test_short_term_off_gives_exactly_the_long_term_score():
    model = tiny_model(use_short_term=False)
    np.testing.assert_allclose(model.score(0), EMBEDDINGS @ model.u_lt[0])
    assert model.attributions(0, 2).size == 0


def test_empty_memory_falls_back_to_long_term():
    model = tiny_model()
    model.memory[0] = []
    np.testing.assert_allclose(model.score(0), EMBEDDINGS @ model.u_lt[0])


def test_memory_keeps_only_the_most_recent_positive_interactions():
    model = tiny_model(memory_size=2)
    model.observe(0, 2, 5.0, 0.0)
    assert model.memory[0] == [1, 2]


def test_one_threshold_decides_memory_in_both_fit_and_observe(dataset, split):
    """The plan's code used 3.0 at initialisation and 3.5 on update."""
    model = CognitiveRecommender(n_factors=4, memory_size=50, positive_threshold=4.0)
    model.fit(split.train, dataset.n_users, dataset.n_items)
    for user, group in split.train.groupby("user"):
        assert model.memory[user] == group.loc[group["rating"] >= 4.0, "item"].tolist()
    before = list(model.memory[0])
    model.observe(0, 0, 3.0, 0.0)
    assert model.memory[0] == before
    model.observe(0, 0, 4.0, 0.0)
    assert model.memory[0] == [*before, 0]


def test_alpha_moves_towards_the_target_using_the_blended_prediction():
    model = tiny_model(rpe_lr=1.0, rpe_center=0.0, rpe_smoothing=0.5)
    reward = 1.0  # a 5-star rating
    surprise = abs(reward - expected_reward_for_item_2())
    target = 0.5 + 1.0 * surprise
    model.observe(0, 2, 5.0, 0.0)
    assert model.alpha[0] == pytest.approx(0.5 + 0.5 * (target - 0.5))


def test_larger_rpe_learning_rate_moves_alpha_further():
    slow, fast = tiny_model(rpe_lr=0.1), tiny_model(rpe_lr=1.0)
    slow.observe(0, 2, 1.0, 0.0)
    fast.observe(0, 2, 1.0, 0.0)
    assert abs(fast.alpha[0] - 0.5) > abs(slow.alpha[0] - 0.5)


def test_rpe_off_keeps_alpha_fixed():
    model = tiny_model(use_rpe=False, rpe_lr=1.0)
    model.observe(0, 2, 1.0, 0.0)
    assert model.alpha[0] == 0.5


def test_alpha_stays_within_its_bounds_under_repeated_surprise():
    model = tiny_model(rpe_lr=100.0, rpe_smoothing=1.0, alpha_min=0.1, alpha_max=0.9)
    model.observe(0, 1, 5.0, 0.0)
    assert model.alpha[0] == pytest.approx(0.9)
    model.u_lt[0] = model._candidates[2]
    model.memory[0] = [2]
    model.observe(0, 2, 5.0, 0.0)
    assert 0.1 <= model.alpha[0] <= 0.9


def test_one_surprising_rating_cannot_fully_redirect_plasticity():
    model = tiny_model(rpe_lr=100.0, rpe_smoothing=0.3)
    model.observe(0, 1, 1.0, 0.0)
    assert 0.1 < model.alpha[0] < 0.9


def test_long_term_memory_stays_unit_length_and_moves_towards_liked_items():
    model = tiny_model(lt_rate=0.5)
    before = model._candidates[1] @ model.u_lt[0]
    model.observe(0, 1, 5.0, 0.0)
    assert np.linalg.norm(model.u_lt[0]) == pytest.approx(1.0)
    assert model._candidates[1] @ model.u_lt[0] > before


def test_normalize_embeddings_flag_changes_candidate_vectors_only(dataset, split):
    args = (split.train, dataset.n_users, dataset.n_items)
    unit = CognitiveRecommender(n_factors=4, normalize_embeddings=True).fit(*args)
    raw = CognitiveRecommender(n_factors=4, normalize_embeddings=False).fit(*args)

    norms = np.linalg.norm(unit._candidates, axis=1)
    assert np.allclose(norms[norms > 0], 1.0)
    np.testing.assert_allclose(raw._candidates, SVD(n_factors=4).fit(*args).item_vecs)
    np.testing.assert_array_equal(unit._unit, raw._unit)
    np.testing.assert_array_equal(unit.u_lt, raw.u_lt)


def test_items_with_no_history_get_a_zero_embedding():
    unit = _unit_rows(np.array([[1.0, 2.0], [0.0, 0.0], [1e-14, 0.0]]))
    np.testing.assert_array_equal(unit[1], [0.0, 0.0])
    np.testing.assert_array_equal(unit[2], [0.0, 0.0])


def test_ucb_bonus_matches_the_formula():
    bonus = ucb_bonus(np.array([0.0, 3.0]), 10.0, 0.5)
    assert bonus[0] == pytest.approx(0.5 * math.sqrt(math.log(11.0) / 1.0))
    assert bonus[1] == pytest.approx(0.5 * math.sqrt(math.log(11.0) / 4.0))


def test_exposure_rises_by_exactly_one_per_list():
    tracker = ExposureTracker(np.array([2.0, 0.0, 1.0]))
    assert tracker.total == 3.0
    tracker.record(np.array([0, 1]))
    np.testing.assert_array_equal(tracker.counts, [3.0, 1.0, 1.0])
    assert tracker.total == 5.0


def test_observing_feedback_does_not_count_as_exposure():
    """The plan's code added a stale recommendation list once per test row."""
    model = tiny_model(exploration_beta=0.1)
    model.observe(0, 2, 5.0, 0.0)
    np.testing.assert_array_equal(model.exposure.counts, [0.0, 0.0, 0.0])


def test_model_d_is_model_c_plus_the_bonus():
    plain = tiny_model(exploration_beta=0.0)
    explorer = tiny_model(exploration_beta=0.2)
    bonus = ucb_bonus(explorer.exposure.counts, explorer.exposure.total, 0.2)
    np.testing.assert_allclose(explorer.score(0), plain.score(0) + bonus)


def test_exposed_items_get_a_smaller_bonus_than_unexposed_items():
    model = tiny_model(exploration_beta=0.5)
    model.record_impression(np.array([2]))
    bonus = model.score(0) - tiny_model(exploration_beta=0.0).score(0)
    assert bonus[2] < bonus[0]
    assert bonus[0] == bonus[1]


def test_memory_diagnostics_detects_recency_only_attention():
    dominated = tiny_model(decay_lambda=100.0)
    report = dominated.memory_diagnostics(0, np.empty(0, dtype=int))
    assert report["recency_share"] == 1.0
    assert tiny_model().memory_diagnostics(0, np.empty(0, dtype=int))["recency_share"] < 1.0
    empty = tiny_model()
    empty.memory[0] = []
    assert empty.memory_diagnostics(0, np.empty(0, dtype=int)) is None
