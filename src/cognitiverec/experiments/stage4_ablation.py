"""Stage 4: ablation study and trade-off curves. Owners: Mudit and Abhinav."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cognitiverec.config import load_config, load_tuned
from cognitiverec.data import Dataset, Split
from cognitiverec.evaluation.protocol import test_phase, validation_phase
from cognitiverec.evaluation.reliability import evaluate_simulated_shift, evaluate_stability
from cognitiverec.experiments.common import (
    ModelSpec,
    StageOutput,
    evaluate_trust,
    load_context,
    paired_comparisons,
    print_report,
    ranking_summary,
    record_cohorts,
    record_models,
    start_run,
)
from cognitiverec.models import CognitiveRecommender, build_model
from cognitiverec.plots import plot_accuracy_coverage, plot_adaptation_stability


def run_sweeps(config: dict[str, Any], dataset: Dataset, split: Split) -> dict[str, pd.DataFrame]:
    """Draw the trade-off curves on validation. Never touches `split.test`."""
    k = config["evaluation"]["k"]
    seeds = config["seeds"]
    settings = config["reliability"]
    phase = validation_phase(dataset, split, config["evaluation"]["positive_threshold"])
    tuned = load_tuned(config, "cognitive")["model_d"]

    beta_rows = []
    for beta in sorted({*config["sweeps"]["beta"], tuned["exploration_beta"]}):
        model = phase.fit(CognitiveRecommender(**{**tuned, "exploration_beta": beta}))
        runs = [ranking_summary(model, phase, k, seed) for seed in seeds]
        beta_rows.append(
            {
                "exploration_beta": beta,
                "ndcg": float(np.mean([run["ndcg"] for run in runs])),
                "coverage_pct": float(np.mean([run["coverage_pct"] for run in runs])),
                "gini": float(np.mean([run["gini"] for run in runs])),
                "rank_merit_gap": float(np.mean([run["rank_merit_gap"] for run in runs])),
            }
        )

    baseline_rows = []
    baselines = load_tuned(config, "baselines")
    for kind, name in (("Popularity", "Popularity"), ("ItemKNN", "ItemKNN"), ("SVD", "SVD")):
        model = phase.fit(build_model(kind, **baselines[kind]))
        summary = ranking_summary(model, phase, k)
        baseline_rows.append(
            {"model": name, "ndcg": summary["ndcg"], "coverage_pct": summary["coverage_pct"]}
        )

    alpha_rows = []
    report_step = settings["shift_report_step"]
    base_alpha = tuned.get("base_alpha", CognitiveRecommender().base_alpha)
    for alpha in sorted({*config["sweeps"]["base_alpha"], base_alpha}):
        model = phase.fit(CognitiveRecommender(**{**tuned, "base_alpha": alpha}))
        stability, adaptation, retention = [], [], []
        for seed in seeds:
            curve = evaluate_simulated_shift(
                model,
                phase,
                settings["shift_steps"],
                k,
                np.random.default_rng(seed),
                settings["shift_min_item_popularity"],
            )
            at_step = curve[curve["step"] == report_step]
            adaptation.append(float(at_step["adaptation"].mean()))
            retention.append(float(at_step["retention_pct"].mean()))
            stability.append(
                evaluate_stability(
                    model, phase, k, settings["stability_steps"], np.random.default_rng(seed)
                )
            )
        alpha_rows.append(
            {
                "base_alpha": alpha,
                "stability": float(np.mean(stability)),
                "shift_adaptation": float(np.mean(adaptation)),
                "shift_retention_pct": float(np.mean(retention)),
            }
        )

    return {
        "beta_sweep": pd.DataFrame(beta_rows),
        "baselines_val": pd.DataFrame(baseline_rows),
        "alpha_sweep": pd.DataFrame(alpha_rows),
    }


def run(config: dict[str, Any] | None = None, dataset: Dataset | None = None) -> StageOutput:
    config = load_config("stage4_ablation") if config is None else config
    context = load_context(config, dataset)
    run_dir = start_run(config, "stage4")
    phase = test_phase(context.dataset, context.split, context.positive_threshold)
    tuned = load_tuned(config, "cognitive")["model_d"]
    models = [
        ModelSpec(ablation["name"], "Cognitive", {**tuned, **ablation["overrides"]})
        for ablation in config["ablations"]
    ]
    record_cohorts(
        run_dir, phase, validation_phase(context.dataset, context.split, context.positive_threshold)
    )
    record_models(run_dir, models)

    rows, per_user = [], []
    for spec in models:
        print(f"Evaluating {spec.name}...", flush=True)
        result = evaluate_trust(spec, phase, config)
        rows.append({"model": spec.name, **result.summary})
        per_user.append(result.per_user)
    tables = {
        "ablation": pd.DataFrame(rows),
        "per_user": pd.concat(per_user, ignore_index=True),
    }
    tables["comparisons"] = paired_comparisons(
        tables["per_user"], config["comparison_reference"], config["comparison_pairs"]
    )

    print("Drawing trade-off curves on validation...", flush=True)
    tables.update(run_sweeps(config, context.dataset, context.split))
    for name, table in tables.items():
        table.to_csv(run_dir / f"{name}.csv", index=False)

    figures = [
        plot_accuracy_coverage(
            tables["beta_sweep"],
            tables["baselines_val"],
            tuned["exploration_beta"],
            run_dir / "tradeoff_accuracy_coverage.png",
        ),
        plot_adaptation_stability(
            tables["alpha_sweep"],
            tuned.get("base_alpha", CognitiveRecommender().base_alpha),
            config["reliability"]["shift_report_step"],
            run_dir / "tradeoff_adaptation_stability.png",
        ),
    ]
    print_report(tables["ablation"])
    print("\n=== Paired differences over users (95% bootstrap interval) ===")
    print(tables["comparisons"].to_string(index=False, float_format="%.4f"))
    print("\n=== Exploration weight on validation ===")
    print(tables["beta_sweep"].to_string(index=False, float_format="%.4f"))
    print("\n=== Base plasticity on validation ===")
    print(tables["alpha_sweep"].to_string(index=False, float_format="%.4f"))
    print(f"\nResults saved to {run_dir}")
    return StageOutput(run_dir, tables, figures)


if __name__ == "__main__":
    run()
