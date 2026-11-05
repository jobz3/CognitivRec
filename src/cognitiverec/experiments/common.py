"""Helpers shared by the stage scripts."""

from __future__ import annotations

import itertools
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cognitiverec.config import PROJECT_ROOT, load_tuned, resolve_path
from cognitiverec.data import Dataset, Split, chronological_split, load_movielens_100k
from cognitiverec.evaluation.explainability import evaluate_fidelity
from cognitiverec.evaluation.fairness import fairness_report
from cognitiverec.evaluation.protocol import Phase, evaluate_ranking
from cognitiverec.evaluation.reliability import (
    evaluate_cold_start,
    evaluate_noise,
    evaluate_replay,
    evaluate_simulated_shift,
    evaluate_stability,
)
from cognitiverec.evaluation.stats import bootstrap_ci
from cognitiverec.models import Recommender, build_model

QUALITY_METRICS = ("precision", "recall", "ndcg", "novelty", "diversity")

# Which result columns belong to which research question.
REPORT_GROUPS = {
    "RQ1 quality": QUALITY_METRICS,
    "RQ2 fairness": (
        "coverage_pct",
        "gini",
        "merit_gap",
        "recall_head",
        "recall_mid",
        "recall_tail",
        "activity_gap",
    ),
    "RQ3 explanation fidelity": (
        "fidelity_gap",
        "fidelity_gap_low",
        "fidelity_gap_high",
        "top1_change_strongest_pct",
        "top1_change_other_pct",
        "score_rises_pct",
        "kendall_tau",
    ),
    "RQ4 reliability": (
        "replay_ndcg",
        "drift_ndcg",
        "no_drift_ndcg",
        "shift_adaptation",
        "shift_adaptation_gain",
        "shift_retention_pct",
        "noise_retention_2_pct",
        "noise_retention_5_pct",
        "stability",
        "cold_ndcg_5",
        "cold_ndcg_10",
        "cold_ndcg_20",
        "cold_ndcg_full",
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    """A named model and the hyperparameters to build it with."""

    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def build(self) -> Recommender:
        return build_model(self.kind, **self.params)


@dataclass(frozen=True)
class Context:
    """Everything a stage needs to know about the data."""

    config: dict[str, Any]
    dataset: Dataset
    split: Split

    @property
    def k(self) -> int:
        return self.config["evaluation"]["k"]

    @property
    def positive_threshold(self) -> float:
        return self.config["evaluation"]["positive_threshold"]


@dataclass
class StageOutput:
    """What a stage returns, so that the notebook can display it."""

    run_dir: Path
    tables: dict[str, pd.DataFrame]
    figures: list[Path] = field(default_factory=list)


def load_context(config: dict[str, Any], dataset: Dataset | None = None) -> Context:
    """Load and split the data. Tests pass their own small `dataset`."""
    dataset = load_movielens_100k(resolve_path(config, "data")) if dataset is None else dataset
    return Context(config, dataset, chronological_split(dataset.interactions, **config["split"]))


def _git_revision() -> str:
    """Return the current commit, marked if the working tree has uncommitted changes."""

    def git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
            )
        except OSError:  # git is not installed; a run should not fail over its own bookkeeping
            return ""
        return result.stdout.strip()

    revision = git("rev-parse", "--short", "HEAD") or "unknown"
    return f"{revision}-dirty" if git("status", "--porcelain") else revision


def start_run(config: dict[str, Any], stage: str) -> Path:
    """Create this run's output folder and record the exact configuration and code version."""
    run_dir = resolve_path(config, "results") / stage / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"git": _git_revision(), "config": config}, handle, sort_keys=True)
    return run_dir


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Return every combination of a hyperparameter grid. An empty grid gives one empty set."""
    names = list(grid)
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*grid.values())]


def standard_models(config: dict[str, Any]) -> list[ModelSpec]:
    """Return models A to D with their tuned hyperparameters."""
    baselines = load_tuned(config, "baselines")
    cognitive = load_tuned(config, "cognitive")
    return [
        ModelSpec("A: Popularity", "Popularity"),
        ModelSpec("B1: ItemKNN", "ItemKNN", baselines["ItemKNN"]),
        ModelSpec("B2: SVD", "SVD", baselines["SVD"]),
        ModelSpec("C: Cognitive", "Cognitive", cognitive["model_c"]),
        ModelSpec("D: Cognitive + UCB", "Cognitive", cognitive["model_d"]),
    ]


def ranking_summary(model: Recommender, phase: Phase, k: int, seed: int | None = None) -> dict:
    """Return the RQ1 and RQ2 metrics of one recommendation pass."""
    rng = None if seed is None else np.random.default_rng(seed)
    run = evaluate_ranking(model, phase, k, rng)
    summary = {metric: float(run.per_user[metric].mean()) for metric in QUALITY_METRICS}
    summary.update(fairness_report(run, phase))
    return summary


@dataclass
class TrustResult:
    """One model's results across all research questions."""

    summary: dict[str, float]
    spread: dict[str, float]
    per_user: pd.DataFrame
    shift_curve: pd.DataFrame


def _seed_results(
    model: Recommender, phase: Phase, config: dict[str, Any], seed: int
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Run every seeded test. Each gets its own generator, so tests cannot affect one another."""
    k = config["evaluation"]["k"]
    settings = config["reliability"]

    def rng() -> np.random.Generator:
        return np.random.default_rng(seed)

    run = evaluate_ranking(model, phase, k, rng())
    metrics = {metric: float(run.per_user[metric].mean()) for metric in QUALITY_METRICS}
    metrics.update(fairness_report(run, phase))

    replay = evaluate_replay(model, phase, k, rng())
    metrics["replay_ndcg"] = float(replay["ndcg"].mean())
    metrics["drift_ndcg"] = float(replay.loc[replay["observed_drift"], "ndcg"].mean())
    metrics["no_drift_ndcg"] = float(replay.loc[~replay["observed_drift"], "ndcg"].mean())

    curve = evaluate_simulated_shift(
        model,
        phase,
        settings["shift_steps"],
        k,
        rng(),
        settings["shift_min_item_popularity"],
    )
    at_step = curve[curve["step"] == settings["shift_report_step"]]
    metrics["shift_adaptation"] = float(at_step["adaptation"].iloc[0])
    # The gain matters: popular items can already fall in the new genre with no adaptation at all.
    metrics["shift_adaptation_gain"] = metrics["shift_adaptation"] - float(
        curve["adaptation"].iloc[0]
    )
    metrics["shift_retention_pct"] = float(at_step["retention_pct"].iloc[0])

    for level in settings["noise_levels"]:
        metrics[f"noise_retention_{level}_pct"] = evaluate_noise(model, phase, level, k, rng())
    metrics["stability"] = evaluate_stability(model, phase, k, settings["stability_steps"], rng())
    # Replay is the only test here in which RPE can act, so its per-user values are kept as well.
    per_user = run.per_user.merge(
        replay[["user", "ndcg"]].rename(columns={"ndcg": "replay_ndcg"}), on="user", how="left"
    )
    return metrics, per_user.assign(seed=seed), curve.assign(seed=seed)


def evaluate_trust(spec: ModelSpec, phase: Phase, config: dict[str, Any]) -> TrustResult:
    """Fit one model and run all RQ1 to RQ4 evaluations on it, averaged over seeds."""
    model = phase.fit(spec.build())
    settings = config["reliability"]
    rows, per_user, curves = [], [], []
    for seed in config["seeds"]:
        metrics, users, curve = _seed_results(model, phase, config, seed)
        if config.get("cold_start", False):
            metrics.update(
                evaluate_cold_start(
                    spec.build,
                    phase,
                    settings["cold_start_sizes"],
                    settings["cold_start_fraction"],
                    config["evaluation"]["k"],
                    np.random.default_rng(seed),
                )
            )
        rows.append(metrics)
        per_user.append(users)
        curves.append(curve)

    table = pd.DataFrame(rows)
    # Fidelity has no random part, so it runs once rather than once per seed.
    fidelity = evaluate_fidelity(model, phase)
    return TrustResult(
        summary={**table.mean().to_dict(), **fidelity},
        spread=table.std(ddof=0).to_dict(),
        per_user=pd.concat(per_user).assign(model=spec.name),
        shift_curve=(
            pd.concat(curves).groupby("step").mean().drop(columns="seed").reset_index()
        ).assign(model=spec.name),
    )


def paired_comparisons(
    per_user: pd.DataFrame,
    reference: str,
    extra_pairs: Iterable[Iterable[str]] = (),
    metrics: Iterable[str] = ("ndcg", "replay_ndcg"),
) -> pd.DataFrame:
    """Compare models user by user, with bootstrap intervals over users.

    Every model is compared with `reference`, plus any `extra_pairs` of (model, baseline). Pairs
    whose models were not run are skipped. An interval that excludes zero is evidence that the
    difference comes from the mechanism rather than from which users happened to be sampled.
    """
    rng = np.random.default_rng(0)
    models = list(per_user["model"].unique())
    rows = []
    for metric in metrics:
        table = per_user.groupby(["user", "model"])[metric].mean().unstack("model")
        pairs = [(model, reference) for model in models if model != reference]
        pairs.extend(tuple(pair) for pair in extra_pairs)
        for model, baseline in pairs:
            if model not in table.columns or baseline not in table.columns:
                continue
            mean, low, high = bootstrap_ci((table[model] - table[baseline]).to_numpy(), rng)
            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "compared_with": baseline,
                    "difference": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def print_report(metrics: pd.DataFrame) -> None:
    """Print a results table, one block per research question."""
    for title, columns in REPORT_GROUPS.items():
        present = [column for column in columns if column in metrics.columns]
        if present:
            print(f"\n=== {title} ===")
            print(metrics[["model", *present]].to_string(index=False, float_format="%.4f"))
