"""Stage 3: evaluate all models on test across RQ1 to RQ4. All owners."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cognitiverec.config import load_config
from cognitiverec.data import Dataset
from cognitiverec.evaluation.protocol import test_phase
from cognitiverec.experiments.common import (
    StageOutput,
    evaluate_trust,
    load_context,
    paired_comparisons,
    print_report,
    ranking_and_replay,
    record_cohorts,
    record_models,
    standard_models,
    start_run,
)
from cognitiverec.plots import plot_shift_curves


def run(config: dict[str, Any] | None = None, dataset: Dataset | None = None) -> StageOutput:
    config = load_config("stage3_trust") if config is None else config
    context = load_context(config, dataset)
    run_dir = start_run(config, "stage3")
    phase = test_phase(context.dataset, context.split, context.positive_threshold)
    models = standard_models(config)
    record_cohorts(run_dir, phase)
    record_models(run_dir, models)

    summaries, spreads, per_user, curves, examples = [], [], [], [], []
    for spec in models:
        print(f"Evaluating {spec.name}...", flush=True)
        result = evaluate_trust(spec, phase, config)
        summaries.append({"model": spec.name, **result.summary})
        spreads.append({"model": spec.name, **result.spread})
        per_user.append(result.per_user)
        curves.append(result.shift_curve)
        examples.extend({"model": spec.name, **row} for row in result.fidelity_examples)

    tables = {
        "metrics": pd.DataFrame(summaries),
        "metrics_std": pd.DataFrame(spreads),
        "shift_curves": pd.concat(curves, ignore_index=True),
        "per_user": pd.concat(per_user, ignore_index=True),
        "fidelity_examples": pd.DataFrame(examples),
    }
    tables["comparisons"] = paired_comparisons(
        tables["per_user"], config["comparison_reference"], config["comparison_pairs"]
    )
    for name, table in tables.items():
        table.to_csv(run_dir / f"{name}.csv", index=False)

    figures = [plot_shift_curves(tables["shift_curves"], run_dir / "shift_curves.png")]
    print_report(tables["metrics"])
    print("\n=== Paired differences over users (95% bootstrap interval) ===")
    print(tables["comparisons"].to_string(index=False, float_format="%.4f"))
    print(f"\nResults saved to {run_dir}")
    return StageOutput(run_dir, tables, figures)


def run_sensitivity(config: dict[str, Any], dataset: Dataset | None = None) -> StageOutput:
    """Repeat the main ranking, fairness and replay endpoints under the alternate split."""
    context = load_context(config, dataset)
    run_dir = start_run(config, "sensitivity")
    phase = test_phase(context.dataset, context.split, context.positive_threshold)
    models = standard_models(config)
    record_cohorts(run_dir, phase)
    record_models(run_dir, models)

    summaries, per_user = [], []
    for spec in models:
        print(f"Checking split sensitivity for {spec.name}...", flush=True)
        model = phase.fit(spec.build())
        results = [ranking_and_replay(model, phase, context.k, seed) for seed in config["seeds"]]
        summaries.append(
            {"model": spec.name, **pd.DataFrame([row for row, _ in results]).mean().to_dict()}
        )
        per_user.extend(frame.assign(model=spec.name) for _, frame in results)
    tables = {
        "metrics": pd.DataFrame(summaries),
        "per_user": pd.concat(per_user, ignore_index=True),
    }
    tables["comparisons"] = paired_comparisons(
        tables["per_user"], config["comparison_reference"], config["comparison_pairs"]
    )
    for name, table in tables.items():
        table.to_csv(run_dir / f"{name}.csv", index=False)
    print_report(tables["metrics"])
    print(f"\nSplit sensitivity saved to {run_dir}")
    return StageOutput(run_dir, tables)


if __name__ == "__main__":
    run()
