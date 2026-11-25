"""Test one exposure policy with both EASE and the cognitive scorer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cognitiverec.config import load_config, resolve_path
from cognitiverec.data import Dataset, chronological_split, load_movielens_100k
from cognitiverec.evaluation.protocol import Phase, evaluate_ranking, test_phase, validation_phase
from cognitiverec.evaluation.stats import bootstrap_ci
from cognitiverec.experiments.common import (
    ModelSpec,
    StageOutput,
    finish_run,
    ranking_summary,
    run_manifest,
    start_run,
    summarize_ranking,
)
from cognitiverec.models.base import Recommender
from cognitiverec.models.exploration import ExposureExploration

SCORERS = {"EASE": "B3: EASE", "Cognitive": "C: Cognitive"}
BASE = "base"
EXPLORE = "exploration"
EFFECT_METRICS = ("ndcg", "coverage_pct", "rank_merit_gap", "recall_tail")


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _score_scale(scorer: Recommender, phase: Phase) -> float:
    """Use only validation candidates to put the same search grid in each scorer's units."""
    spreads = []
    for user in phase.users:
        scores = scorer.score(int(user)).copy()
        scores[phase.excluded_items(int(user))] = np.nan
        candidates = scores[np.isfinite(scores)]
        if len(candidates):
            spreads.append(candidates.std())
    scale = float(np.median(spreads)) if spreads else 0.0
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Exploration tuning requires a positive validation candidate-score scale.")
    return scale


def tune_policy(scorer: Recommender, phase: Phase, config: dict) -> pd.DataFrame:
    """Select the best coverage among feasible strengths, preferring weaker ties."""
    settings = config["exploration_study"]
    k = config["evaluation"]["k"]
    scale = _score_scale(scorer, phase)
    rows = []
    for strength in sorted({0.0, *settings["multipliers"]}):
        beta = strength * scale
        model = ExposureExploration(scorer, phase.popularity, beta)
        summary = pd.DataFrame(
            [ranking_summary(model, phase, k, seed) for seed in config["seeds"]]
        ).mean()
        rows.append(
            {
                "multiplier": strength,
                "beta": beta,
                "score_scale": scale,
                **{metric: summary[metric] for metric in EFFECT_METRICS},
            }
        )
    grid = pd.DataFrame(rows)
    baseline = grid.loc[grid["multiplier"] == 0, "ndcg"].iloc[0]
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError("A relative NDCG loss budget needs positive validation NDCG at beta zero.")
    grid["ndcg_loss_pct"] = 100 * (1 - grid["ndcg"] / baseline)
    grid["feasible"] = grid["ndcg"] >= baseline * (1 - settings["max_ndcg_loss"])
    selected = (
        grid.loc[grid["feasible"]]
        .sort_values(["coverage_pct", "multiplier"], ascending=[False, True])
        .index[0]
    )
    return grid.assign(selected=grid.index == selected)


def _evaluate_cell(
    scorer: Recommender, phase: Phase, beta: float, config: dict, identity: dict
) -> tuple[list[dict], list[pd.DataFrame]]:
    model = ExposureExploration(scorer, phase.popularity, beta)
    summaries, users = [], []
    for seed in config["seeds"]:
        labels = {**identity, "seed": seed}
        result = evaluate_ranking(
            model, phase, config["evaluation"]["k"], np.random.default_rng(seed)
        )
        summaries.append({**labels, **summarize_ranking(result, phase)})
        users.append(result.per_user.assign(**labels))
    return summaries, users


def _paired_effects(metrics: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Report paired accuracy effects and the difference between the two policy effects."""
    rng = np.random.default_rng(0)
    rows = []
    for window, subset in users.groupby("window", sort=False):
        individual = (
            subset.groupby(["user", "scorer", "policy"])["ndcg"]
            .mean()
            .unstack(["scorer", "policy"])
        )
        average = metrics.loc[metrics["window"] == window].set_index(["scorer", "policy"])[
            list(EFFECT_METRICS)
        ]
        ndcg_deltas, aggregate_deltas = {}, {}
        for scorer in SCORERS:
            ndcg_deltas[scorer] = individual[scorer, EXPLORE] - individual[scorer, BASE]
            aggregate_deltas[scorer] = average.loc[scorer, EXPLORE] - average.loc[scorer, BASE]
        ndcg_deltas["Interaction"] = ndcg_deltas["Cognitive"] - ndcg_deltas["EASE"]
        aggregate_deltas["Interaction"] = aggregate_deltas["Cognitive"] - aggregate_deltas["EASE"]
        for scorer, deltas in ndcg_deltas.items():
            mean, low, high = bootstrap_ci(deltas.to_numpy(), rng)
            rows.append(
                {
                    "window": window,
                    "scorer": scorer,
                    "ndcg": mean,
                    "ndcg_ci_low": low,
                    "ndcg_ci_high": high,
                    "paired_users": int(deltas.notna().sum()),
                    **{metric: aggregate_deltas[scorer][metric] for metric in EFFECT_METRICS[1:]},
                }
            )
    return pd.DataFrame(rows)


def run(
    source_run: str | Path, *, quick: bool = False, dataset: Dataset | None = None
) -> StageOutput:
    """Hold the previously selected scorers fixed and retune both exploration policies."""
    source = Path(source_run).resolve(strict=True)
    source_manifest = _read_yaml(source / "manifest.yaml")
    config = _read_yaml(source / "config.yaml")["config"]
    config["exploration_study"] = load_config("exploration_study")["exploration_study"]
    config["source_rolling_run"] = str(source)
    if quick:
        config["seeds"] = config["seeds"][:1]
        config["execution_mode"] = "smoke"
        if config["paths"]["results"] == "results":
            config["paths"]["results"] = "results/exploration-smoke"

    data_dir = resolve_path(config, "data") if dataset is None else None
    if dataset is None:
        dataset = load_movielens_100k(data_dir)
    run_dir = start_run(config, "exploration_study")
    manifest = run_manifest(run_dir, data_dir, config["execution_mode"])
    if manifest["dataset_sha256"] != source_manifest["dataset_sha256"]:
        raise ValueError("The dataset must match the completed rolling source run.")
    manifest["source_rolling_run"] = source_manifest["run_id"]
    manifest["source_rolling_code_sha256"] = source_manifest["code_sha256"]
    grids, summaries, users = [], [], []
    for window in config["windows"]:
        name = window["name"]
        models = _read_yaml(source / name / "models.yaml")
        split = chronological_split(
            dataset.interactions,
            window["train"],
            window["val"],
            mode="global",
            test_end=window["test_end"],
        )
        threshold = config["evaluation"]["positive_threshold"]
        validation = validation_phase(dataset, split, threshold)
        final = test_phase(dataset, split, threshold)
        for label, model in SCORERS.items():
            spec = ModelSpec(model, **models[model])
            print(f"{name}: validating exploration for {label}", flush=True)
            grid = tune_policy(validation.fit(spec.build()), validation, config)
            grids.append(grid.assign(window=name, scorer=label))
            beta = float(grid.loc[grid["selected"], "beta"].iloc[0])
            fitted = final.fit(spec.build())
            for policy, weight in ((BASE, 0.0), (EXPLORE, beta)):
                rows, frames = _evaluate_cell(
                    fitted,
                    final,
                    weight,
                    config,
                    {"window": name, "scorer": label, "policy": policy, "beta": weight},
                )
                summaries.extend(rows)
                users.extend(frames)

    seed_metrics = pd.DataFrame(summaries)
    metrics = (
        seed_metrics.drop(columns="seed")
        .groupby(["window", "scorer", "policy"], sort=False, as_index=False)
        .mean()
    )
    per_user = pd.concat(users, ignore_index=True)
    tables = {
        "metrics": metrics,
        "effects": _paired_effects(metrics, per_user),
        "validation": pd.concat(grids, ignore_index=True),
        "per_user": per_user,
        "seed_metrics": seed_metrics,
        "windows": pd.read_csv(source / "windows.csv"),
    }
    for name, table in tables.items():
        table.to_csv(run_dir / f"{name}.csv", index=False)
    finish_run(run_dir, manifest)
    print(f"Exploration comparison saved to {run_dir}")
    return StageOutput(run_dir, tables)
