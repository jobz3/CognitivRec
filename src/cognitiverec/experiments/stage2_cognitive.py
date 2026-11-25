"""Stage 2: tune Models C and D on validation. Owners: Mudit and Abhinav."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yaml

from cognitiverec.config import load_config, save_tuned
from cognitiverec.data import Dataset, Split
from cognitiverec.evaluation.protocol import Phase, evaluate_ranking, validation_phase
from cognitiverec.evaluation.reliability import evaluate_replay
from cognitiverec.experiments.common import (
    StageOutput,
    load_context,
    ranking_summary,
    record_cohorts,
    start_run,
)
from cognitiverec.models import CognitiveRecommender


def _sample_trials(search: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the model's defaults, as a baseline trial, plus distinct random grid picks."""
    rng = np.random.default_rng(search["seed"])
    grid = search["grid"]
    defaults = CognitiveRecommender()
    trials: list[dict[str, Any]] = [{name: getattr(defaults, name) for name in grid}]
    seen = {tuple(trials[0].values())}
    for _ in range(search["n_trials"] * 20):
        if len(trials) > search["n_trials"]:
            break
        trial = {name: values[int(rng.integers(len(values)))] for name, values in grid.items()}
        key = tuple(trial.values())
        if key not in seen:
            seen.add(key)
            trials.append(trial)
    return trials


def _search_model_c(config: dict[str, Any], phase: Phase) -> tuple[dict[str, Any], pd.DataFrame]:
    """Random search over Model C's hyperparameters, on validation only."""
    search = config["search"]
    k = config["evaluation"]["k"]
    threshold = config["evaluation"]["positive_threshold"]
    rows = []
    best_params: dict[str, Any] = {}
    best_value = -1.0
    for trial in _sample_trials(search):
        params = {**trial, "positive_threshold": threshold, "exploration_beta": 0.0}
        model = phase.fit(CognitiveRecommender(**params))
        metrics = {
            "ndcg": float(evaluate_ranking(model, phase, k).per_user["ndcg"].mean()),
            "replay_ndcg": float(evaluate_replay(model, phase, k)["ndcg"].mean()),
        }
        rows.append({**params, **metrics})
        if metrics[search["objective"]] > best_value:
            best_value, best_params = metrics[search["objective"]], params
    if not best_params:
        raise ValueError(f"No eligible validation users for {search['objective']}.")
    return best_params, pd.DataFrame(rows)


def _select_beta(
    config: dict[str, Any], phase: Phase, model_c: dict[str, Any]
) -> tuple[float, pd.DataFrame]:
    """Choose the highest coverage whose NDCG loss against Model C stays within the budget."""
    selection = config["beta_selection"]
    k = config["evaluation"]["k"]
    rows = []
    for beta in sorted({0.0, *selection["grid"]}):
        model = phase.fit(CognitiveRecommender(**{**model_c, "exploration_beta": beta}))
        # The order users are served in affects exposure counts, so average over seeds.
        runs = [ranking_summary(model, phase, k, seed) for seed in config["seeds"]]
        rows.append(
            {
                "exploration_beta": beta,
                "ndcg": float(np.mean([run["ndcg"] for run in runs])),
                "coverage_pct": float(np.mean([run["coverage_pct"] for run in runs])),
                "gini": float(np.mean([run["gini"] for run in runs])),
                "rank_merit_gap": float(np.mean([run["rank_merit_gap"] for run in runs])),
            }
        )
    table = pd.DataFrame(rows)
    floor = (1.0 - selection["max_ndcg_loss"]) * table.loc[table["exploration_beta"] == 0.0, "ndcg"]
    allowed = table[table["ndcg"] >= floor.iloc[0]]
    chosen = allowed.sort_values(["coverage_pct", "exploration_beta"], ascending=[False, True])
    return float(chosen["exploration_beta"].iloc[0]), table


def memory_report(model: CognitiveRecommender, phase: Phase) -> dict[str, float]:
    """Summarise whether short-term memory reaches the ranking, or sits unused."""
    rows = [
        model.memory_diagnostics(int(user), phase.excluded_items(int(user))) for user in phase.users
    ]
    rows = [row for row in rows if row is not None]
    long_term = float(np.mean([row["long_term_spread"] for row in rows]))
    short_term = float(np.mean([row["short_term_spread"] for row in rows]))
    return {
        "short_to_long_term_spread": short_term / long_term if long_term > 0 else float("nan"),
        "attention_on_newest_memory_pct": float(
            100 * np.mean([row["recency_share"] for row in rows])
        ),
    }


def tune_cognitive(
    config: dict[str, Any], dataset: Dataset, split: Split
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    """Tune Model C, then Model D's beta on top of it. Never touches `split.test`."""
    phase = validation_phase(dataset, split, config["evaluation"]["positive_threshold"])
    model_c, search_table = _search_model_c(config, phase)
    beta, beta_table = _select_beta(config, phase, model_c)
    params = {"model_c": model_c, "model_d": {**model_c, "exploration_beta": beta}}
    return params, {"search": search_table, "beta": beta_table}


def run(config: dict[str, Any] | None = None, dataset: Dataset | None = None) -> StageOutput:
    config = load_config("stage2_cognitive") if config is None else config
    context = load_context(config, dataset)
    run_dir = start_run(config, "stage2")

    print("Tuning Models C and D on validation...", flush=True)
    params, tables = tune_cognitive(config, context.dataset, context.split)
    tuned_path = save_tuned(config, "cognitive", params)
    tables["search"].to_csv(run_dir / "search.csv", index=False)
    tables["beta"].to_csv(run_dir / "beta_sweep.csv", index=False)

    phase = validation_phase(context.dataset, context.split, context.positive_threshold)
    record_cohorts(run_dir, phase)
    diagnostics = memory_report(phase.fit(CognitiveRecommender(**params["model_c"])), phase)
    with (run_dir / "memory_report.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(diagnostics, handle)

    print(f"  Model C: {params['model_c']}")
    print(f"  Model D: exploration_beta = {params['model_d']['exploration_beta']}")
    print(f"  written to {tuned_path}")
    print("\nDoes short-term memory reach the ranking?")
    for name, value in diagnostics.items():
        print(f"  {name}: {value:.3f}")
    print("\n=== Exploration weight on validation ===")
    print(tables["beta"].to_string(index=False, float_format="%.4f"))
    print(f"\nResults saved to {run_dir}")
    return StageOutput(run_dir, tables)


if __name__ == "__main__":
    run()
