"""Each stage must run end to end, and tuning must never read test data."""

import dataclasses

import pandas as pd
import pytest
import yaml

from cognitiverec.config import load_config, load_tuned
from cognitiverec.experiments import (
    common,
    stage1_baselines,
    stage2_cognitive,
    stage3_trust,
    stage4_ablation,
)
from cognitiverec.experiments.common import expand_grid, standard_models


@pytest.fixture
def overrides(tmp_path):
    """Small settings for the synthetic dataset, writing everything to a temporary folder."""
    return {
        "seeds": [1, 2],
        "paths": {"results": str(tmp_path / "results"), "tuned": str(tmp_path / "tuned")},
        "evaluation": {"k": 5},
        "reliability": {
            "shift_steps": [1, 2],
            "shift_report_step": 2,
            "shift_min_item_popularity": 1,
            "stability_steps": 2,
            "cold_start_sizes": [5],
        },
        "baselines": {
            "Popularity": {},
            "ItemKNN": {"top_k": [5, 10], "shrinkage": [1.0]},
            "SVD": {"n_factors": [2, 4]},
        },
        "search": {
            "n_trials": 3,
            "grid": {
                "n_factors": [2, 4],
                "memory_size": [3, 6],
                "base_alpha": [0.3, 0.6],
                "decay_lambda": [0.1],
                "attention_temperature": [1.0],
                "rpe_lr": [0.2],
                "lt_rate": [0.05],
                "normalize_embeddings": [True, False],
            },
        },
        "beta_selection": {"grid": [0.01, 0.1]},
        "sweeps": {"beta": [0.0, 0.1], "base_alpha": [0.3, 0.6]},
    }


def stage_config(stage: str, overrides: dict) -> dict:
    config = load_config(stage, overrides)
    # Grids are replaced outright, rather than merged with the real project grids.
    config["search"]["grid"] = overrides["search"]["grid"]
    config["baselines"] = overrides["baselines"]
    if stage == "stage4_ablation":
        config["sweeps"] = overrides["sweeps"]
        config["ablations"] = config["ablations"][:2]
    return config


def run_tuning_stages(overrides, dataset):
    stage1_baselines.run(stage_config("stage1_baselines", overrides), dataset)
    stage2_cognitive.run(stage_config("stage2_cognitive", overrides), dataset)


def test_expand_grid():
    assert expand_grid({}) == [{}]
    assert expand_grid({"a": [1, 2], "b": [3]}) == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def test_tuning_stages_never_read_test_data(overrides, dataset, split):
    """With no test split present at all, any access to it would raise an error."""
    no_test = dataclasses.replace(split, test=None)
    config = stage_config("stage1_baselines", overrides)
    best, _ = stage1_baselines.tune_baselines(config, dataset, no_test)
    assert set(best) == {"Popularity", "ItemKNN", "SVD"}

    config = stage_config("stage2_cognitive", overrides)
    params, _ = stage2_cognitive.tune_cognitive(config, dataset, no_test)
    assert set(params) == {"model_c", "model_d"}


def test_trade_off_curves_never_read_test_data(overrides, dataset, split):
    run_tuning_stages(overrides, dataset)
    config = stage_config("stage4_ablation", overrides)
    tables = stage4_ablation.run_sweeps(config, dataset, dataclasses.replace(split, test=None))
    assert set(tables) == {"beta_sweep", "baselines_val", "alpha_sweep"}


def test_stage1_writes_results_and_tuned_parameters(overrides, dataset):
    output = stage1_baselines.run(stage_config("stage1_baselines", overrides), dataset)
    assert (output.run_dir / "baselines_test.csv").exists()
    assert (output.run_dir / "temporal_report.yaml").exists()
    tuned = load_tuned(stage_config("stage1_baselines", overrides), "baselines")
    assert tuned["ItemKNN"]["top_k"] in (5, 10)
    assert output.tables["test"]["ndcg"].between(0, 1).all()


def test_runs_record_their_configuration(overrides, dataset):
    output = stage1_baselines.run(stage_config("stage1_baselines", overrides), dataset)
    snapshot = yaml.safe_load((output.run_dir / "config.yaml").read_text())
    assert snapshot["config"]["seeds"] == [1, 2]
    assert snapshot["git"]


def test_runs_survive_a_machine_without_git(overrides, monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(common.subprocess, "run", no_git)
    run_dir = common.start_run(stage_config("stage1_baselines", overrides), "stage1")
    assert yaml.safe_load((run_dir / "config.yaml").read_text())["git"] == "unknown"


def test_later_stages_fail_clearly_without_tuned_parameters(overrides):
    with pytest.raises(FileNotFoundError, match="make stage1 stage2"):
        standard_models(load_config("stage3_trust", overrides))


def test_stage2_beta_stays_within_the_ndcg_budget(overrides, dataset):
    config = stage_config("stage2_cognitive", overrides)
    output = stage2_cognitive.run(config, dataset)
    beta = load_tuned(config, "cognitive")["model_d"]["exploration_beta"]
    table = output.tables["beta"]
    budget = 0.9 * table.loc[table["exploration_beta"] == 0.0, "ndcg"].iloc[0]
    assert table.loc[table["exploration_beta"] == beta, "ndcg"].iloc[0] >= budget


def test_stage3_measures_every_model_on_every_question(overrides, dataset):
    run_tuning_stages(overrides, dataset)
    output = stage3_trust.run(stage_config("stage3_trust", overrides), dataset)
    metrics = output.tables["metrics"].set_index("model")
    assert len(metrics) == 5
    assert (output.run_dir / "shift_curves.png").exists()
    assert (output.run_dir / "per_user.csv").exists()

    # Nothing is hard-coded: fidelity is NaN only for the models with no attributions.
    assert metrics["fidelity_gap"].isna().tolist() == [True, False, True, False, False]
    measured = ["ndcg", "coverage_pct", "replay_ndcg", "stability", "noise_retention_2_pct"]
    assert metrics[measured].notna().all().all()
    assert metrics["cold_ndcg_5"].notna().all()


def test_stage3_compares_models_on_paired_users(overrides, dataset):
    run_tuning_stages(overrides, dataset)
    output = stage3_trust.run(stage_config("stage3_trust", overrides), dataset)
    comparisons = output.tables["comparisons"]
    assert set(comparisons["metric"]) == {"ndcg", "replay_ndcg"}
    assert "D: Cognitive + UCB" in comparisons["model"].tolist()
    assert "C: Cognitive" in comparisons["compared_with"].tolist()
    assert (comparisons["ci_low"] <= comparisons["ci_high"]).all()


def test_paired_comparison_reports_a_known_difference():
    per_user = pd.DataFrame(
        {
            "user": [0, 1, 0, 1],
            "model": ["a", "a", "b", "b"],
            "ndcg": [0.5, 0.7, 0.2, 0.4],
        }
    )
    row = common.paired_comparisons(per_user, "b", metrics=("ndcg",)).iloc[0]
    assert row["difference"] == pytest.approx(0.3)


def test_paired_comparison_skips_models_that_were_not_run():
    per_user = pd.DataFrame({"user": [0, 0], "model": ["a", "b"], "ndcg": [0.5, 0.2]})
    table = common.paired_comparisons(per_user, "b", [("a", "missing")], metrics=("ndcg",))
    assert table["compared_with"].tolist() == ["b"]


def test_paired_comparison_ignores_users_with_no_value_for_a_metric():
    per_user = pd.DataFrame(
        {
            "user": [0, 1, 0, 1],
            "model": ["a", "a", "b", "b"],
            "replay_ndcg": [0.5, float("nan"), 0.2, float("nan")],
        }
    )
    row = common.paired_comparisons(per_user, "b", metrics=("replay_ndcg",)).iloc[0]
    assert row["difference"] == pytest.approx(0.3)


def test_stage4_runs_ablations_and_draws_trade_offs(overrides, dataset):
    run_tuning_stages(overrides, dataset)
    output = stage4_ablation.run(stage_config("stage4_ablation", overrides), dataset)
    assert len(output.tables["ablation"]) == 2
    assert set(output.tables["comparisons"]["compared_with"]) == {"Full Model D"}
    assert len(output.figures) == 2
    assert all(path.exists() for path in output.figures)
