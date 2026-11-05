import numpy as np
import pandas as pd
import pytest

from cognitiverec.models import SVD, ItemKNN, Popularity


def small_history():
    rows = [
        (0, 0, 5.0, 1),
        (0, 1, 3.0, 2),
        (1, 0, 4.0, 1),
        (1, 1, 1.0, 2),
        (1, 2, 2.0, 3),
        (2, 1, 2.0, 1),
        (2, 2, 5.0, 2),
    ]
    return pd.DataFrame(rows, columns=["user", "item", "rating", "timestamp"])


def test_popularity_scores_are_counts_and_ignore_feedback():
    model = Popularity().fit(small_history(), 3, 4)
    np.testing.assert_array_equal(model.score(0), [2, 3, 2, 0])
    model.observe(0, 3, 5.0, 9)
    np.testing.assert_array_equal(model.score(0), [2, 3, 2, 0])


def test_item_knn_matches_the_written_formula():
    history = small_history()
    model = ItemKNN(top_k=4, shrinkage=2.0).fit(history, 3, 4)

    ratings = np.zeros((3, 4))
    ratings[history["user"], history["item"]] = history["rating"]
    normalised = ratings / (np.sqrt((ratings**2).sum(axis=0)) + 1e-8)
    presence = (ratings > 0).astype(float)
    co_counts = presence.T @ presence
    weights = (normalised.T @ normalised) * (co_counts / (co_counts + 2.0))
    np.fill_diagonal(weights, 0.0)

    np.testing.assert_allclose(model.score(0), ratings[0] @ weights)


def test_item_knn_keeps_only_top_k_neighbours():
    model = ItemKNN(top_k=1, shrinkage=0.0).fit(small_history(), 3, 4)
    assert (model.neighbours_of.toarray() > 0).sum(axis=1).max() == 1


def test_item_knn_handles_zero_shrinkage_without_nan():
    model = ItemKNN(top_k=4, shrinkage=0.0).fit(small_history(), 3, 4)
    assert np.isfinite(model.score(0)).all()


def test_item_knn_attributions_add_up_to_the_score():
    model = ItemKNN(top_k=4, shrinkage=1.0).fit(small_history(), 3, 4)
    for item in range(4):
        assert model.attributions(1, item).sum() == pytest.approx(model.score(1)[item])


def test_item_knn_deletion_lowers_the_score_by_the_attribution():
    model = ItemKNN(top_k=4, shrinkage=1.0).fit(small_history(), 3, 4)
    attributions = model.attributions(1, 3)
    base = model.score(1)
    for entry, attribution in enumerate(attributions):
        assert base[3] - model.score_without(1, entry)[3] == pytest.approx(attribution)


def test_item_knn_folds_in_feedback_without_refitting():
    model = ItemKNN(top_k=4, shrinkage=1.0).fit(small_history(), 3, 4)
    before = model.score(0).copy()
    weights = model.neighbours_of.copy()
    model.observe(0, 2, 5.0, 9)
    assert not np.array_equal(before, model.score(0))
    assert (weights != model.neighbours_of).nnz == 0


def test_item_knn_binary_mode_ignores_rating_values():
    history = small_history()
    lowered = history.assign(rating=1.0)
    first = ItemKNN(binary=True, shrinkage=1.0).fit(history, 3, 4)
    second = ItemKNN(binary=True, shrinkage=1.0).fit(lowered, 3, 4)
    np.testing.assert_allclose(first.score(0), second.score(0))


def test_svd_is_reproducible(dataset, split):
    first = SVD(n_factors=4, seed=1).fit(split.train, dataset.n_users, dataset.n_items)
    second = SVD(n_factors=4, seed=1).fit(split.train, dataset.n_users, dataset.n_items)
    np.testing.assert_array_equal(first.item_vecs, second.item_vecs)


def test_svd_factors_are_in_descending_order(dataset, split):
    model = SVD(n_factors=4).fit(split.train, dataset.n_users, dataset.n_items)
    assert np.all(np.diff(model.singular) <= 0)


def test_svd_fold_in_of_an_existing_rating_changes_nothing(dataset, split):
    """Re-projecting a user from their own ratings must land on the vector the SVD already found."""
    model = SVD(n_factors=4).fit(split.train, dataset.n_users, dataset.n_items)
    row = split.train.iloc[0]
    before = model.score(int(row["user"])).copy()
    model.observe(int(row["user"]), int(row["item"]), float(row["rating"]), 0.0)
    np.testing.assert_allclose(model.score(int(row["user"])), before, atol=1e-8)


def test_svd_fold_in_of_a_new_rating_changes_scores(dataset, split):
    model = SVD(n_factors=4).fit(split.train, dataset.n_users, dataset.n_items)
    user = 0
    unseen = np.setdiff1d(
        np.arange(dataset.n_items), split.train.loc[split.train["user"] == user, "item"]
    )
    before = model.score(user).copy()
    model.observe(user, int(unseen[0]), 5.0, 0.0)
    assert not np.allclose(before, model.score(user))
