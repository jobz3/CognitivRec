import numpy as np
import pandas as pd
import pytest

from cognitiverec.data import Dataset, chronological_split
from cognitiverec.evaluation.protocol import Phase, rank_top_k, validation_phase
from cognitiverec.evaluation.reliability import (
    evaluate_cold_start,
    evaluate_noise,
    evaluate_replay,
    evaluate_simulated_shift,
    evaluate_stability,
    observed_drift_users,
)
from cognitiverec.models import CognitiveRecommender, ItemKNN, Popularity


def rng():
    return np.random.default_rng(7)


def test_zero_noise_retention_is_exactly_100_for_every_model(factory, val_phase):
    """The plan's noise test failed this: Model D scored 0.8% with no noise involved."""
    model = val_phase.fit(factory())
    assert evaluate_noise(model, val_phase, 0, 5, rng()) == 100.0


def test_noise_reaches_an_adaptive_model(val_phase):
    model = val_phase.fit(CognitiveRecommender(n_factors=4, memory_size=4))
    assert evaluate_noise(model, val_phase, 5, 5, rng()) != 100.0


def test_every_model_receives_the_same_noise(val_phase):
    class Logged:
        log: dict[str, list] = {}

        def observe(self, user, item, rating, timestamp):
            Logged.log.setdefault(type(self).__name__, []).append((user, item, rating))

    class LoggedPopularity(Logged, Popularity):
        pass

    class LoggedKNN(Logged, ItemKNN):
        pass

    for model in (LoggedPopularity(), LoggedKNN(top_k=10, shrinkage=1.0)):
        evaluate_noise(val_phase.fit(model), val_phase, 2, 5, rng())
    assert len(Logged.log["LoggedPopularity"]) == 2 * len(val_phase.users)
    assert Logged.log["LoggedPopularity"] == Logged.log["LoggedKNN"]


def test_noise_avoids_the_users_history_and_targets(val_phase):
    class Recording(Popularity):
        seen: list = []

        def observe(self, user, item, rating, timestamp):
            Recording.seen.append((user, item))

    evaluate_noise(val_phase.fit(Recording()), val_phase, 3, 5, rng())
    assert Recording.seen
    for user, item in Recording.seen:
        assert item not in val_phase.seen_items(user)
        assert item not in val_phase.streams[user]["item"].to_numpy()


def test_replay_scores_every_model_on_the_same_targets(val_phase):
    frames = []
    for build in (Popularity, lambda: ItemKNN(top_k=10), lambda: CognitiveRecommender(n_factors=4)):
        frames.append(evaluate_replay(val_phase.fit(build()), val_phase, 5, rng()))
    assert len(frames[0]) > 0
    for frame in frames[1:]:
        pd.testing.assert_frame_equal(
            frames[0][["user", "n_targets", "observed_drift"]],
            frame[["user", "n_targets", "observed_drift"]],
        )


def test_replay_never_recommends_what_was_just_replayed(val_phase):
    seen_lists = []

    class Capturing(Popularity):
        def record_impression(self, items):
            seen_lists.append(items.copy())

    model = val_phase.fit(Capturing())
    evaluate_replay(model, val_phase, 5)
    user = int(evaluate_replay(model, val_phase, 5)["user"].iloc[0])
    replayed = val_phase.streams[user].iloc[: (len(val_phase.streams[user]) + 1) // 2]["item"]
    exclude = np.concatenate([val_phase.seen_items(user), replayed.to_numpy()])
    items = rank_top_k(model.score(user), exclude, 5)
    assert not set(items.tolist()) & set(replayed.tolist())
    assert any(np.array_equal(items, listed) for listed in seen_lists)


def test_replay_feedback_changes_an_adaptive_model_but_not_the_callers_copy(val_phase):
    model = val_phase.fit(CognitiveRecommender(n_factors=4, memory_size=4))
    before = model.score(0).copy()
    evaluate_replay(model, val_phase, 5)
    np.testing.assert_array_equal(before, model.score(0))


def test_replay_keeps_timestamp_ties_together_and_matches_frozen_popularity(val_phase, dataset):
    phase = Phase(
        "tied",
        val_phase.history,
        val_phase.targets.assign(timestamp=2_000, rating=5.0),
        dataset,
    )
    assert evaluate_replay(phase.fit(Popularity()), phase, 5).empty
    replay = evaluate_replay(val_phase.fit(Popularity()), val_phase, 5)
    assert len(replay) > 0
    np.testing.assert_array_equal(replay["ndcg"], replay["frozen_ndcg"])


def drift_dataset() -> Dataset:
    """User 0 moves from genre Alpha to genre Beta; user 1 stays with Alpha throughout."""
    genres = np.zeros((40, 3), dtype=np.int8)
    genres[:20, 1] = 1
    genres[20:, 2] = 1
    rows = []
    for position, item in enumerate([*range(14), *range(20, 26)]):
        rows.append((0, item, 5.0, position))
    for position, item in enumerate(range(20)):
        rows.append((1, item, 5.0, position))
    return Dataset(
        interactions=pd.DataFrame(rows, columns=["user", "item", "rating", "timestamp"]),
        item_genres=genres,
        item_titles=tuple(str(index) for index in range(40)),
        genre_names=("unknown", "Alpha", "Beta"),
        n_users=2,
        n_items=40,
    )


def test_observed_drift_flags_a_change_of_top_genre():
    data = drift_dataset()
    phase = validation_phase(data, chronological_split(data.interactions, train=0.7, val=0.3))
    assert observed_drift_users(phase) == {0}


def test_shift_starts_at_100_and_adapts_an_adaptive_model(val_phase):
    model = val_phase.fit(CognitiveRecommender(n_factors=4, memory_size=4, base_alpha=0.8))
    curve = evaluate_simulated_shift(
        model, val_phase, steps=(1, 3), k=5, rng=rng(), min_item_popularity=1
    )
    assert curve["step"].tolist() == [0, 1, 3]
    assert curve["retention_pct"].iloc[0] == pytest.approx(100.0)
    assert curve["users"].iloc[0] > 0
    assert curve["adaptation"].iloc[-1] > curve["adaptation"].iloc[0]


def test_shift_gives_every_model_the_same_injected_items(val_phase):
    class Logged:
        log: dict[str, list] = {}

        def observe(self, user, item, rating, timestamp):
            Logged.log.setdefault(type(self).__name__, []).append((user, item))

    class LoggedPopularity(Logged, Popularity):
        pass

    class LoggedKNN(Logged, ItemKNN):
        pass

    for model in (LoggedPopularity(), LoggedKNN(top_k=10)):
        evaluate_simulated_shift(
            val_phase.fit(model), val_phase, steps=(1, 3), k=5, rng=rng(), min_item_popularity=1
        )
    assert Logged.log["LoggedPopularity"]
    assert Logged.log["LoggedPopularity"] == Logged.log["LoggedKNN"]


def test_shift_feedback_gain_is_zero_for_frozen_popularity(val_phase):
    curve = evaluate_simulated_shift(
        val_phase.fit(Popularity()),
        val_phase,
        steps=(1, 3),
        k=5,
        rng=rng(),
        min_item_popularity=1,
    )
    assert len(curve) == 3
    assert curve["feedback_gain"].eq(0).all()


def test_shift_skips_users_when_no_genre_has_enough_items(val_phase):
    model = val_phase.fit(Popularity())
    curve = evaluate_simulated_shift(
        model, val_phase, steps=(1,), k=5, rng=rng(), min_item_popularity=10_000
    )
    assert curve.empty


def test_stability_is_a_valid_overlap_and_popularity_barely_moves(val_phase):
    still = evaluate_stability(val_phase.fit(Popularity()), val_phase, 5, 3, rng())
    assert 0.5 <= still <= 1.0
    adaptive = val_phase.fit(CognitiveRecommender(n_factors=4, memory_size=4))
    assert 0.0 <= evaluate_stability(adaptive, val_phase, 5, 3, rng()) <= 1.0


def test_stability_ignores_the_item_just_consumed(val_phase):
    """A model that only loses the consumed item from its list has not changed its mind."""

    class FixedOrder(Popularity):
        def score(self, user):
            return -np.arange(len(self.counts), dtype=float)

    value = evaluate_stability(val_phase.fit(FixedOrder()), val_phase, 5, 1)
    # Each step scores 1.0, or 4/5 if a listed item was consumed. Without removing that item from
    # the earlier list, those steps would score 4/6 and pull the mean below this bound.
    assert value < 1.0
    assert value >= 4 / 5


def test_cold_start_refits_with_truncated_histories(val_phase):
    fits = []

    class Spying(Popularity):
        def fit(self, history, n_users, n_items):
            fits.append(history.copy())
            return super().fit(history, n_users, n_items)

    results = evaluate_cold_start(Spying, val_phase, sizes=(5,), fraction=0.25, k=5, rng=rng())
    assert set(results) == {"cold_ndcg_full", "cold_ndcg_5"}
    full, truncated = fits
    sizes = truncated.groupby("user").size()
    cut_users = sizes[sizes == 5].index
    assert len(cut_users) == max(1, int(0.25 * len(val_phase.users)))
    for user in cut_users:
        expected = full.loc[full["user"] == user, "item"].head(5).tolist()
        assert truncated.loc[truncated["user"] == user, "item"].tolist() == expected


def test_cold_start_uses_the_normal_protocol_so_exposure_keeps_counting(val_phase):
    """Skipping impressions here would freeze Model D's exposure, as in the plan's noise test."""

    class Counting(Popularity):
        impressions = 0

        def record_impression(self, items):
            Counting.impressions += 1

    evaluate_cold_start(Counting, val_phase, sizes=(5,), fraction=0.25, k=5, rng=rng())
    assert Counting.impressions == 2 * len(val_phase.users)  # the full fit, and one cut size


def test_cold_start_picks_the_same_users_for_every_model(val_phase):
    def users_cut_for(build):
        fits = []

        class Spying(build):
            def fit(self, history, n_users, n_items):
                fits.append(history)
                return super().fit(history, n_users, n_items)

        evaluate_cold_start(Spying, val_phase, sizes=(5,), fraction=0.25, k=5, rng=rng())
        sizes = fits[1].groupby("user").size()
        return sizes[sizes == 5].index.tolist()

    assert users_cut_for(Popularity) == users_cut_for(ItemKNN)


def test_cold_start_gives_a_valid_ndcg_for_an_adaptive_model(val_phase):
    results = evaluate_cold_start(
        lambda: CognitiveRecommender(n_factors=4, memory_size=4),
        val_phase,
        sizes=(5,),
        fraction=0.5,
        k=5,
        rng=rng(),
    )
    assert 0.0 <= results["cold_ndcg_5"] <= 1.0
