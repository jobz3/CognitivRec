"""Models may only ever learn from a phase's history."""

import dataclasses

import numpy as np
import pandas as pd

from cognitiverec.evaluation.protocol import test_phase, validation_phase


def test_validation_history_is_exactly_train(dataset, split):
    phase = validation_phase(dataset, split)
    pd.testing.assert_frame_equal(phase.history, split.train)


def test_final_history_is_train_plus_val_in_time_order(dataset, split):
    phase = test_phase(dataset, split)
    assert len(phase.history) == len(split.train) + len(split.val)
    for _, group in phase.history.groupby("user"):
        assert group["timestamp"].is_monotonic_increasing


def test_changing_test_rows_leaves_fitted_state_unchanged(factory, dataset, split):
    original = test_phase(dataset, split)
    corrupted_test = split.test.copy()
    corrupted_test["rating"] = 6.0 - corrupted_test["rating"]
    corrupted_test["item"] = (corrupted_test["item"] + 1) % dataset.n_items
    corrupted = test_phase(dataset, dataclasses.replace(split, test=corrupted_test))

    first = original.fit(factory())
    second = corrupted.fit(factory())
    for user in range(dataset.n_users):
        np.testing.assert_array_equal(first.score(user), second.score(user))


def test_changing_val_rows_does_not_affect_tuning_fit(factory, dataset, split):
    corrupted_val = split.val.copy()
    corrupted_val["rating"] = 6.0 - corrupted_val["rating"]
    first = validation_phase(dataset, split).fit(factory())
    second = validation_phase(dataset, dataclasses.replace(split, val=corrupted_val)).fit(factory())
    for user in range(dataset.n_users):
        np.testing.assert_array_equal(first.score(user), second.score(user))
