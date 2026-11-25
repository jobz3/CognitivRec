"""Stage 1: tune the baselines on validation, then report them on test. Owner: Jobin."""

from __future__ import annotations

from typing import Any

import pandas as pd
import yaml

from cognitiverec.config import load_config, save_tuned
from cognitiverec.data import Dataset, Split, temporal_report
from cognitiverec.evaluation.protocol import evaluate_ranking, test_phase, validation_phase
from cognitiverec.experiments.common import (
    StageOutput,
    expand_grid,
    load_context,
    print_report,
    ranking_summary,
    record_cohorts,
    start_run,
)
from cognitiverec.models import build_model


def tune_baselines(
    config: dict[str, Any], dataset: Dataset, split: Split
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Pick each baseline's best hyperparameters by validation NDCG. Never touches `split.test`."""
    evaluation = config["evaluation"]
    phase = validation_phase(dataset, split, evaluation["positive_threshold"])
    if not len(phase.users):
        raise ValueError("No users have eligible positive validation targets.")
    rows = []
    best: dict[str, dict[str, Any]] = {}
    for kind, grid in config["baselines"].items():
        best_ndcg = -1.0
        for params in expand_grid(grid):
            model = phase.fit(build_model(kind, **params))
            ndcg = float(evaluate_ranking(model, phase, evaluation["k"]).per_user["ndcg"].mean())
            rows.append({"model": kind, "params": params, "val_ndcg": ndcg})
            if ndcg > best_ndcg:
                best_ndcg, best[kind] = ndcg, params
    return best, pd.DataFrame(rows)


def run(config: dict[str, Any] | None = None, dataset: Dataset | None = None) -> StageOutput:
    config = load_config("stage1_baselines") if config is None else config
    context = load_context(config, dataset)
    run_dir = start_run(config, "stage1")

    report = temporal_report(context.split)
    print("Temporal properties of the split:")
    for name, value in report.items():
        print(f"  {name}: {value:.2f}")
    with (run_dir / "temporal_report.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle)

    print("\nTuning baselines on validation...", flush=True)
    best, tuning = tune_baselines(config, context.dataset, context.split)
    tuning.to_csv(run_dir / "tuning.csv", index=False)
    tuned_path = save_tuned(config, "baselines", best)
    for kind, params in best.items():
        print(f"  {kind}: {params}")
    print(f"  written to {tuned_path}")

    threshold, k = context.positive_threshold, context.k
    val = validation_phase(context.dataset, context.split, threshold)
    final = test_phase(context.dataset, context.split, threshold)
    record_cohorts(run_dir, val, final)
    rows = {"val": [], "test": []}
    for kind, params in best.items():
        rows["val"].append(
            {"model": kind, **ranking_summary(val.fit(build_model(kind, **params)), val, k)}
        )
        rows["test"].append(
            {"model": kind, **ranking_summary(final.fit(build_model(kind, **params)), final, k)}
        )

    tables = {phase: pd.DataFrame(phase_rows) for phase, phase_rows in rows.items()}
    tables["tuning"] = tuning
    for phase in ("val", "test"):
        tables[phase].to_csv(run_dir / f"baselines_{phase}.csv", index=False)
    print("\n##### Baselines on the test split #####")
    print_report(tables["test"])
    print(f"\nResults saved to {run_dir}")
    return StageOutput(run_dir, tables)


if __name__ == "__main__":
    run()
