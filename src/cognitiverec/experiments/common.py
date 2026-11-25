"""Helpers shared by the stage scripts."""

from __future__ import annotations

import hashlib
import itertools
import platform
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cognitiverec.config import PROJECT_ROOT, load_tuned, resolve_path
from cognitiverec.data import Dataset, Split, chronological_split, load_movielens_100k
from cognitiverec.data.movielens import FILES
from cognitiverec.evaluation.explainability import evaluate_fidelity
from cognitiverec.evaluation.fairness import fairness_report
from cognitiverec.evaluation.protocol import Phase, RankingRun, evaluate_ranking
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
        "rank_merit_gap",
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
        "frozen_replay_ndcg",
        "replay_gain",
        "replay_users",
        "drift_ndcg",
        "no_drift_ndcg",
        "shift_adaptation",
        "shift_adaptation_gain",
        "shift_feedback_gain",
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%fZ")
    run_dir = resolve_path(config, "results") / stage / stamp
    run_dir.mkdir(parents=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"git": _git_revision(), "config": config}, handle, sort_keys=True)
    return run_dir


def record_cohorts(run_dir: Path, *phases: Phase) -> None:
    """Persist who and what each result can represent."""
    with (run_dir / "cohorts.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({phase.name: phase.cohort for phase in phases}, handle, sort_keys=True)


def record_models(run_dir: Path, models: list[ModelSpec]) -> None:
    with (run_dir / "models.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {spec.name: {"kind": spec.kind, "params": spec.params} for spec in models}, handle
        )


def run_manifest(run_dir: Path, data_dir: Path | None, execution_mode: str) -> dict:
    """Fingerprint data and package sources, including uncommitted source edits."""
    source = hashlib.sha256()
    for path in sorted((PROJECT_ROOT / "src/cognitiverec").rglob("*.py")):
        source.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
        source.update(b"\0")
        source.update(path.read_bytes())
        source.update(b"\0")
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "execution_mode": execution_mode,
        "code_sha256": source.hexdigest(),
        "lock_sha256": hashlib.sha256((PROJECT_ROOT / "uv.lock").read_bytes()).hexdigest(),
        "dataset_sha256": {
            name: hashlib.sha256((data_dir / "ml-100k" / name).read_bytes()).hexdigest()
            for name in FILES
        }
        if data_dir is not None
        else {},
        "versions": {
            "python": platform.python_version(),
            **{package: version(package) for package in ("numpy", "pandas", "scipy", "matplotlib")},
        },
    }


def finish_run(run_dir: Path, manifest: dict) -> None:
    with (run_dir / "manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {**manifest, "completed_utc": datetime.now(timezone.utc).isoformat()}, handle
        )


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
    return summarize_ranking(evaluate_ranking(model, phase, k, rng), phase)


def summarize_ranking(run: RankingRun, phase: Phase) -> dict:
    """Summarize a ranking once so experiments can also retain individual user scores."""
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
    fidelity_examples: list[dict]


def ranking_and_replay(
    model: Recommender, phase: Phase, k: int, seed: int
) -> tuple[dict[str, float], pd.DataFrame]:
    """Measure the same one-shot ranking and replay outcomes under either split."""
    run = evaluate_ranking(model, phase, k, np.random.default_rng(seed))
    metrics = summarize_ranking(run, phase)

    replay = evaluate_replay(model, phase, k, np.random.default_rng(seed))
    metrics["replay_ndcg"] = float(replay["ndcg"].mean())
    metrics["frozen_replay_ndcg"] = float(replay["frozen_ndcg"].mean())
    metrics["replay_gain"] = metrics["replay_ndcg"] - metrics["frozen_replay_ndcg"]
    metrics["replay_users"] = len(replay)
    metrics["drift_ndcg"] = float(replay.loc[replay["observed_drift"], "ndcg"].mean())
    metrics["no_drift_ndcg"] = float(replay.loc[~replay["observed_drift"], "ndcg"].mean())

    replay = replay.assign(replay_gain=replay["ndcg"] - replay["frozen_ndcg"])
    per_user = run.per_user.merge(
        replay[["user", "ndcg", "replay_gain"]].rename(columns={"ndcg": "replay_ndcg"}),
        on="user",
        how="left",
    )
    return metrics, per_user.assign(seed=seed)


def reliability_stress(
    model: Recommender, phase: Phase, config: dict[str, Any], seed: int
) -> tuple[dict[str, float], pd.DataFrame]:
    """Run matched shift, noise and stability tests on the fitted model."""
    k = config["evaluation"]["k"]
    settings = config["reliability"]
    metrics = {}

    def rng() -> np.random.Generator:
        return np.random.default_rng(seed)

    curve = evaluate_simulated_shift(
        model,
        phase,
        settings["shift_steps"],
        k,
        rng(),
        settings["shift_min_item_popularity"],
    )
    at_step = curve[curve["step"] == settings["shift_report_step"]]
    metrics["shift_adaptation"] = float(at_step["adaptation"].mean())
    # The gain matters: popular items can already fall in the new genre with no adaptation at all.
    metrics["shift_adaptation_gain"] = metrics["shift_adaptation"] - float(
        curve.loc[curve["step"] == 0, "adaptation"].mean()
    )
    metrics["shift_retention_pct"] = float(at_step["retention_pct"].mean())
    metrics["shift_feedback_gain"] = float(at_step["feedback_gain"].mean())

    for level in settings["noise_levels"]:
        metrics[f"noise_retention_{level}_pct"] = evaluate_noise(model, phase, level, k, rng())
    metrics["stability"] = evaluate_stability(model, phase, k, settings["stability_steps"], rng())
    return metrics, curve.assign(seed=seed)


def evaluate_trust(spec: ModelSpec, phase: Phase, config: dict[str, Any]) -> TrustResult:
    """Fit one model and run all RQ1 to RQ4 evaluations on it, averaged over seeds."""
    model = phase.fit(spec.build())
    settings = config["reliability"]
    rows, per_user, curves = [], [], []
    for seed in config["seeds"]:
        metrics, users = ranking_and_replay(model, phase, config["evaluation"]["k"], seed)
        stress, curve = reliability_stress(model, phase, config, seed)
        metrics.update(stress)
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
    # Deletion is exhaustive. Use the first ranking seed to fix Model D's decision state.
    examples: list[dict] = []
    fidelity = evaluate_fidelity(
        model, phase, config["evaluation"]["k"], np.random.default_rng(config["seeds"][0]), examples
    )
    return TrustResult(
        summary={**table.mean().to_dict(), **fidelity},
        spread=table.std(ddof=0).to_dict(),
        per_user=pd.concat(per_user).assign(model=spec.name),
        shift_curve=(
            pd.concat(curves).groupby("step").mean().drop(columns="seed").reset_index()
        ).assign(model=spec.name),
        fidelity_examples=examples,
    )


def paired_comparisons(
    per_user: pd.DataFrame,
    reference: str,
    extra_pairs: Iterable[Iterable[str]] = (),
    metrics: Iterable[str] = ("ndcg", "replay_ndcg", "replay_gain"),
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
        if metric not in per_user:
            continue
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
                    "paired_users": int((table[model].notna() & table[baseline].notna()).sum()),
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
