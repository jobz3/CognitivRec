"""Run the benchmark and chronological follow-up experiments."""

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument("--split", choices=("global", "per_user"), default="global")
    experiment.add_argument(
        "--followup", action="store_true", help="Run the rolling temporal study"
    )
    experiment.add_argument(
        "--exploration-study",
        type=Path,
        metavar="ROLLING_RUN",
        help="Compare exploration policies using a completed rolling study",
    )
    parser.add_argument("--quick", action="store_true", help="Smoke run with one seed and 3 trials")
    args = parser.parse_args()

    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"

    from cognitiverec.config import load_config
    from cognitiverec.experiments import (
        exploration_study,
        stage1_baselines,
        stage2_cognitive,
        stage3_trust,
        stage4_ablation,
        temporal_followup,
    )

    if args.exploration_study:
        exploration_study.run(args.exploration_study, quick=args.quick)
        return

    if args.followup:
        temporal_followup.run(temporal_followup.study_config(quick=args.quick))
        return

    overrides = {"split": {"mode": args.split}}
    if args.quick:
        overrides.update(seeds=[42])
    if args.quick or args.split != "global":
        label = f"{args.split}{'-quick' if args.quick else ''}"
        overrides["paths"] = {"results": f"results/{label}", "tuned": f"results/{label}/tuned"}

    stages = (
        (stage1_baselines.run, "stage1_baselines"),
        (stage2_cognitive.run, "stage2_cognitive"),
    )
    if args.split == "per_user":
        stages += ((stage3_trust.run_sensitivity, "stage3_trust"),)
    else:
        stages += (
            (stage3_trust.run, "stage3_trust"),
            (stage4_ablation.run, "stage4_ablation"),
        )
    for stage, name in stages:
        config = load_config(name, overrides)
        if args.quick and name == "stage2_cognitive":
            config["search"]["n_trials"] = 3
        stage(config)


if __name__ == "__main__":
    main()
