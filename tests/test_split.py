import pandas as pd

from cognitiverec.data import chronological_split, temporal_report


def frame(rows):
    return pd.DataFrame(rows, columns=["user", "item", "rating", "timestamp"])


def test_every_user_is_chronological_across_splits(split):
    for user in split.train["user"].unique():
        train = split.train.loc[split.train["user"] == user, "timestamp"]
        val = split.val.loc[split.val["user"] == user, "timestamp"]
        test = split.test.loc[split.test["user"] == user, "timestamp"]
        assert train.max() <= val.min() <= val.max() <= test.min()


def test_no_row_is_lost_or_duplicated(dataset, split):
    combined = pd.concat([split.train, split.val, split.test])
    columns = ["user", "item", "rating", "timestamp"]
    expected = dataset.interactions.sort_values(columns).reset_index(drop=True)
    actual = combined[columns].sort_values(columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False)


def test_twenty_ratings_split_into_fourteen_three_three():
    """0.70 + 0.15 is slightly off 0.85 in floating point, which would make val one row short."""
    result = chronological_split(frame([(0, item, 4.0, item) for item in range(20)]))
    assert (len(result.train), len(result.val), len(result.test)) == (14, 3, 3)


def test_shared_timestamps_keep_dataset_order():
    rows = [(0, item, 4.0, 100) for item in [7, 3, 9, 1, 5, 8, 2, 6, 4, 0]]
    result = chronological_split(frame(rows))
    ordered = pd.concat([result.train, result.val, result.test])["item"].tolist()
    assert ordered == [7, 3, 9, 1, 5, 8, 2, 6, 4, 0]


def test_splitting_twice_gives_the_same_result(dataset, split):
    again = chronological_split(dataset.interactions)
    pd.testing.assert_frame_equal(split.test, again.test)


def test_temporal_report_detects_a_single_sitting():
    rows = [(0, item, 4.0, 1_000 + item) for item in range(20)]
    report = temporal_report(chronological_split(frame(rows)))
    assert report["single_session_users_pct"] == 100.0
    assert report["same_second_ratings_pct"] == 0.0


def test_temporal_report_measures_later_training_rows():
    """Half of all training rows are later than user 0's test rows, and none than user 1's."""
    early = [(0, item, 4.0, item) for item in range(20)]
    late = [(1, item, 4.0, 100_000 + item) for item in range(20)]
    report = temporal_report(chronological_split(frame(early + late)))
    assert report["train_later_than_test_row_pct"] == 25.0
