import numpy as np

from cognitiverec.evaluation.protocol import evaluate_ranking, rank_top_k, user_order
from cognitiverec.models import Popularity


def test_top_k_excludes_masked_items():
    scores = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    np.testing.assert_array_equal(rank_top_k(scores, np.array([0, 2]), 2), [1, 3])


def test_top_k_returns_fewer_rather_than_a_masked_item():
    """Even when too few candidates remain, a masked item must never fill the list."""
    scores = np.array([9.0, 8.0, 7.0])
    np.testing.assert_array_equal(rank_top_k(scores, np.array([0, 1]), 5), [2])


def test_top_k_ties_go_to_the_lower_item_id():
    np.testing.assert_array_equal(rank_top_k(np.ones(4), np.empty(0, dtype=int), 2), [0, 1])


def test_top_k_does_not_change_the_scores_it_is_given():
    scores = np.array([1.0, 2.0])
    rank_top_k(scores, np.array([1]), 1)
    np.testing.assert_array_equal(scores, [1.0, 2.0])


def test_every_model_returns_one_score_per_item_without_changing_state(factory, val_phase):
    model = val_phase.fit(factory())
    first = model.score(0)
    assert first.shape == (val_phase.n_items,)
    np.testing.assert_array_equal(first, model.score(0))


def test_no_model_ever_recommends_a_masked_item(factory, val_phase):
    model = val_phase.fit(factory())
    run = evaluate_ranking(model, val_phase, k=5)
    for user, items in run.recommendations.items():
        assert not set(items.tolist()) & set(val_phase.seen_items(user).tolist())
        assert len(items) == 5


def test_final_phase_masks_validation_items_too(factory, final_phase, split):
    model = final_phase.fit(factory())
    run = evaluate_ranking(model, final_phase, k=5)
    for user, items in run.recommendations.items():
        validation_items = set(split.val.loc[split.val["user"] == user, "item"].tolist())
        assert not set(items.tolist()) & validation_items


def test_evaluation_does_not_change_the_callers_model(factory, val_phase):
    model = val_phase.fit(factory())
    before = model.score(0)
    evaluate_ranking(model, val_phase, k=5, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(before, model.score(0))


def test_each_user_produces_exactly_one_impression(val_phase):
    class Counting(Popularity):
        calls = 0

        def record_impression(self, items):
            Counting.calls += 1

    evaluate_ranking(val_phase.fit(Counting()), val_phase, k=5)
    assert Counting.calls == len(val_phase.users)


def test_user_order_is_seeded_and_is_a_permutation():
    users = np.arange(20)
    first = user_order(users, np.random.default_rng(1))
    second = user_order(users, np.random.default_rng(1))
    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == users.tolist()
    assert first.tolist() != users.tolist()
    assert user_order(users, None) is users
