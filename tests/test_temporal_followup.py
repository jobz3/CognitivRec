"""Regression checks for the added models and rolling study."""

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from cognitiverec.data import chronological_split
from cognitiverec.experiments import exploration_study, paper_figures, temporal_followup
from cognitiverec.models.cognitive import CognitiveRecommender
from cognitiverec.models.ease import EASE
from cognitiverec.models.exploration import ExposureExploration
from cognitiverec.models.latent import rating_to_reward
from cognitiverec.models.recency import RecencyProfile


def test_ease_matches_constrained_ridge_and_folds_in_only_positive_feedback():
    history = pd.DataFrame(
        {"user": [0, 0, 1, 1, 2], "item": [0, 1, 0, 2, 2], "rating": [5, 4, 5, 5, 2]}
    )
    model = EASE(regularization=2).fit(history, 3, 3)
    for item in range(3):
        neighbours = np.arange(3) != item
        features = model.profiles[:, neighbours]
        weights = np.linalg.solve(
            features.T @ features + 2 * np.eye(2), features.T @ model.profiles[:, item]
        )
        np.testing.assert_allclose(model.weights[neighbours, item], weights)
        assert model.weights[item, item] == 0
    model.observe(2, 1, 2, 0)
    np.testing.assert_array_equal(model.score(2), np.zeros(3))
    model.observe(2, 1, 5, 1)
    np.testing.assert_array_equal(model.score(2), model.weights[1])


def test_recency_profile_replay_matches_direct_exponential_weighting(dataset, split):
    model = RecencyProfile(4, False, half_life=2).fit(split.train, dataset.n_users, dataset.n_items)
    user, item, rating = 0, 0, 5
    rows = split.train[split.train["user"] == user]
    items = np.append(rows["item"], item)
    weights = rating_to_reward(np.append(rows["rating"], rating))
    expected = (weights * 2 ** (-np.arange(len(items) - 1, -1, -1) / 2)) @ model.unit[items]
    model.observe(user, item, rating, 0)
    np.testing.assert_allclose(model.profiles[user], expected)


def test_shared_policy_matches_model_d_and_zero_weight_leaves_scores_unchanged(val_phase):
    base = val_phase.fit(CognitiveRecommender(n_factors=4))
    d = val_phase.fit(CognitiveRecommender(n_factors=4, exploration_beta=0.1))
    shared = ExposureExploration(base, val_phase.popularity, beta=0.1)
    zero = ExposureExploration(base, val_phase.popularity, beta=0.0)
    for model in (d, shared, zero):
        model.record_impression(np.array([0, 2, 4]))
    np.testing.assert_allclose(shared.score(0), d.score(0))
    np.testing.assert_array_equal(zero.score(0), base.score(0))


def test_rolling_test_windows_are_disjoint_and_have_no_split_timestamp_ties(dataset):
    windows = temporal_followup.study_config()["windows"]
    splits = [
        chronological_split(
            dataset.interactions, row["train"], row["val"], mode="global", test_end=row["test_end"]
        )
        for row in windows
    ]
    for split in splits:
        assert split.train["timestamp"].max() < split.val["timestamp"].min()
        assert split.val["timestamp"].max() < split.test["timestamp"].min()
    for earlier, later in zip(splits, splits[1:], strict=False):
        assert earlier.test["timestamp"].max() < later.test["timestamp"].min()


def test_followup_records_boundaries_scores_and_manifest(dataset, tmp_path):
    config = temporal_followup.study_config(quick=True)
    config["paths"]["results"] = str(tmp_path)
    config["windows"] = config["windows"][-1:]
    config["search"]["n_trials"] = 0
    output = temporal_followup.run(config, dataset)
    assert len(output.tables["metrics"]) == 8
    assert len(output.tables["seed_metrics"]) == 8
    assert len(output.tables["anchor_reliability"]) == 8
    assert output.tables["windows"].iloc[0]["test_first_utc"].endswith("+00:00")
    assert (output.run_dir / "late" / "models.yaml").is_file()
    manifest = yaml.safe_load((output.run_dir / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["execution_mode"] == "smoke"
    assert manifest["started_utc"] < manifest["completed_utc"]
    assert len(manifest["code_sha256"]) == 64

    result = exploration_study.run(output.run_dir, dataset=dataset)
    assert len(result.tables["metrics"]) == 4
    assert len(result.tables["effects"]) == 3
    chosen = result.tables["validation"].query("selected")
    assert len(chosen) == 2
    assert chosen["feasible"].all()
    assert (chosen["ndcg_loss_pct"] <= 10.0 + 1e-8).all()
    for name, row in (
        result.tables["metrics"].query("policy == 'base'").set_index("scorer").iterrows()
    ):
        original = output.tables["metrics"].set_index("model")
        np.testing.assert_allclose(
            row["ndcg"], original.loc[exploration_study.SCORERS[name], "ndcg"]
        )
    rendered = paper_figures.render(result.run_dir, output.run_dir, result.run_dir / "figures")
    assert len(rendered) == 4
    for path in rendered:
        assert "<svg " in path.with_suffix(".svg").read_text(encoding="utf-8")
        with Image.open(path) as image:
            assert abs(image.info["dpi"][0] - 300) < 0.1
