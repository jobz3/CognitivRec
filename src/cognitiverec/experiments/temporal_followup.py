"""Rolling global evaluation with a stronger collaborative baseline and a recency control."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cognitiverec.config import deep_merge, load_config, resolve_path
from cognitiverec.data import Dataset, Split, chronological_split, load_movielens_100k
from cognitiverec.evaluation.explainability import evaluate_fidelity
from cognitiverec.evaluation.protocol import Phase, test_phase, validation_phase
from cognitiverec.evaluation.reliability import evaluate_replay
from cognitiverec.experiments.common import (
    ModelSpec,
    StageOutput,
    finish_run,
    paired_comparisons,
    ranking_and_replay,
    record_cohorts,
    record_models,
    reliability_stress,
    run_manifest,
    start_run,
)
from cognitiverec.experiments.stage1_baselines import tune_baselines
from cognitiverec.experiments.stage2_cognitive import tune_cognitive


def study_config(quick: bool = False) -> dict:
    """Use the project defaults for cognitive tuning and one study-specific search budget."""
    config = deep_merge(load_config("stage2_cognitive"), load_config("temporal_followup"))
    config["execution_mode"] = "smoke" if quick else "full"
    if quick:
        config["seeds"] = [42]
        config["search"]["n_trials"] = 3
        config["baselines"]["ItemKNN"] = {
            "top_k": [50, 150],
            "shrinkage": [0.0],
            "binary": [False],
            "tfidf": [True],
        }
        config["paths"]["results"] = "results/followup-smoke"
    config["baselines"]["EASE"]["positive_threshold"] = [config["evaluation"]["positive_threshold"]]
    return config


def _select_recency(
    config: dict, phase: Phase, cognitive: dict, k: int
) -> tuple[dict, pd.DataFrame]:
    """Match the representation to C and choose half-life by the same replay objective."""
    shared = {key: cognitive[key] for key in ("n_factors", "normalize_embeddings")}
    rows = []
    for half_life in config["recency_half_lives"]:
        params = {**shared, "half_life": half_life}
        model = phase.fit(ModelSpec("", "RecencyProfile", params).build())
        replay_ndcg = evaluate_replay(model, phase, k)["ndcg"].mean()
        rows.append({"half_life": half_life, "val_replay_ndcg": replay_ndcg})
    tuning = pd.DataFrame(rows)
    best = tuning.sort_values("val_replay_ndcg", ascending=False, kind="stable").iloc[0]
    return {**shared, "half_life": float(best["half_life"])}, tuning


def _window_metadata(name: str, split: Split, phase: Phase) -> dict:
    row = {"window": name, **phase.cohort}
    row["returning_users"] = int(
        split.test.loc[split.test["user"].isin(phase.seen), "user"].nunique()
    )
    for label, frame in (("train", split.train), ("validation", split.val), ("test", split.test)):
        row[f"{label}_rows"] = len(frame)
        for endpoint, value in (
            ("first", frame["timestamp"].min()),
            ("last", frame["timestamp"].max()),
        ):
            row[f"{label}_{endpoint}_utc"] = pd.Timestamp(value, unit="s", tz="UTC").isoformat()
    return row


def _tune_window(
    config: dict, dataset: Dataset, split: Split, validation: Phase, directory: Path
) -> list[ModelSpec]:
    """Select the models using only the preceding validation window and save all trials."""
    k = config["evaluation"]["k"]
    baselines, baseline_trials = tune_baselines(config, dataset, split)
    cognitive, trials = tune_cognitive(config, dataset, split)
    recency, recency_trials = _select_recency(config, validation, cognitive["model_c"], k)
    for label, table in {**trials, "baselines": baseline_trials, "recency": recency_trials}.items():
        table.to_csv(directory / f"{label}_validation.csv", index=False)

    specs = [
        ModelSpec("A: Popularity", "Popularity"),
        ModelSpec("B1: ItemKNN", "ItemKNN", baselines["ItemKNN"]),
        ModelSpec("B2: SVD", "SVD", baselines["SVD"]),
        ModelSpec("B3: EASE", "EASE", baselines["EASE"]),
        ModelSpec("B4: All-history profile", "RecencyProfile", {**recency, "half_life": None}),
        ModelSpec("B5: Recency profile", "RecencyProfile", recency),
        ModelSpec("C: Cognitive", "Cognitive", cognitive["model_c"]),
        ModelSpec("D: Cognitive + UCB", "Cognitive", cognitive["model_d"]),
    ]
    record_models(directory, specs)
    return specs


def _score_window(
    name: str, final: Phase, specs: list[ModelSpec], config: dict, examples: list[dict] | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate shared endpoints, adding stress tests and demo cases at the latest cutoff."""
    k = config["evaluation"]["k"]
    seed_rows, users, stress_rows = [], [], []
    for spec in specs:
        print(f"  scoring {spec.name}", flush=True)
        model = final.fit(spec.build())
        for seed in config["seeds"]:
            identity = {"window": name, "model": spec.name, "seed": seed}
            scores, per_user = ranking_and_replay(model, final, k, seed)
            seed_rows.append({**identity, **scores})
            users.append(per_user.assign(window=name, model=spec.name))
            if examples is not None:
                stress, _ = reliability_stress(model, final, config, seed)
                stress_rows.append({**identity, **stress})
        if examples is not None and spec.name == "D: Cognitive + UCB":
            evaluate_fidelity(
                model,
                final,
                k,
                np.random.default_rng(config["seeds"][0]),
                examples,
                demo_cases=True,
            )
    return pd.DataFrame(seed_rows), pd.concat(users, ignore_index=True), pd.DataFrame(stress_rows)


def run(config: dict | None = None, dataset: Dataset | None = None) -> StageOutput:
    config = study_config() if config is None else config
    data_dir = resolve_path(config, "data") if dataset is None else None
    if dataset is None:
        dataset = load_movielens_100k(data_dir)
    run_dir = start_run(config, "temporal_followup")
    manifest = run_manifest(run_dir, data_dir, config["execution_mode"])
    threshold = config["evaluation"]["positive_threshold"]
    windows, seed_metrics, users, comparisons, examples = [], [], [], [], []
    for window in config["windows"]:
        name = window["name"]
        split = chronological_split(
            dataset.interactions,
            window["train"],
            window["val"],
            mode="global",
            test_end=window["test_end"],
        )
        validation = validation_phase(dataset, split, threshold)
        final = test_phase(dataset, split, threshold)
        windows.append(_window_metadata(name, split, final))
        window_dir = run_dir / name
        window_dir.mkdir()
        record_cohorts(window_dir, validation, final)
        print(
            f"{name}: tuning on {len(validation.users)} users before {len(final.users)} test users",
            flush=True,
        )
        specs = _tune_window(config, dataset, split, validation, window_dir)
        is_last = name == config["windows"][-1]["name"]
        scores, window_users, stress = _score_window(
            name, final, specs, config, examples if is_last else None
        )
        seed_metrics.append(scores)
        users.append(window_users)
        comparisons.append(
            paired_comparisons(
                window_users, config["comparison_reference"], config["comparison_pairs"]
            ).assign(window=name)
        )

    all_seeds = pd.concat(seed_metrics, ignore_index=True)
    tables = {
        "windows": pd.DataFrame(windows),
        "metrics": all_seeds.drop(columns="seed")
        .groupby(["window", "model"], sort=False)
        .mean()
        .reset_index(),
        "seed_metrics": all_seeds,
        "comparisons": pd.concat(comparisons, ignore_index=True),
        "per_user": pd.concat(users, ignore_index=True),
        "demo_examples": pd.DataFrame(examples),
        "anchor_reliability": stress.drop(columns="seed")
        .groupby(["window", "model"], sort=False)
        .mean()
        .reset_index(),
        "anchor_reliability_by_seed": stress,
    }
    for label, table in tables.items():
        table.to_csv(run_dir / f"{label}.csv", index=False)
    finish_run(run_dir, manifest)
    print(f"Follow-up artifacts written to {run_dir}")
    return StageOutput(run_dir, tables)
