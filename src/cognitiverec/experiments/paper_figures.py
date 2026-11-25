"""Render the paper and slide figure set from completed experiment tables."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

from cognitiverec import plots
from cognitiverec.plots import (
    INK_MUTED,
    INK_SECONDARY,
    SERIES_COLOR,
    SURFACE,
    _line,
    _new_figure,
    _save,
    _title,
    plot_adaptation_stability,
    plot_shift_curves,
)

SCORER_STYLES = {"EASE": ("#00785d", "s"), "Cognitive": ("#1565a0", "o")}


def _legend(figure, handles, labels, columns: int = 2) -> None:
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=columns,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_SECONDARY,
    )


def _window_labels(windows: pd.DataFrame) -> list[str]:
    return [f"{row.window.title()}\n(n = {row.ranking_users})" for row in windows.itertuples()]


def _interval(axis, x, y, low, high, *, horizontal: bool = False, color=SERIES_COLOR) -> None:
    """Draw paired percentile intervals and their means."""
    if horizontal:
        axis.hlines(y, low, high, color=color, linewidth=1.4)
    else:
        axis.vlines(x, low, high, color=color, linewidth=1.4)
    axis.scatter(x, y, color=color, s=32, edgecolor=SURFACE, linewidth=0.7, zorder=3)


def _ranking(
    metrics: pd.DataFrame, paired: pd.DataFrame, windows: pd.DataFrame, path: Path
) -> Path:
    figure, (left, right) = _new_figure(8.8, 3.75, columns=2)
    labels = _window_labels(windows)
    x = np.arange(len(labels))
    for (scorer, policy), group in metrics.groupby(["scorer", "policy"], sort=False):
        color, marker = SCORER_STYLES[scorer]
        exploring = policy == "exploration"
        name = f"{scorer} + exploration" if exploring else scorer
        values = group.set_index("window").loc[windows["window"], "ndcg"]
        _line(
            left,
            x + (0.035 if exploring else -0.035),
            values,
            color,
            marker,
            name,
            linestyle="--" if exploring else "-",
        )
    _title(left, "Ranking quality across test windows")
    left.set_ylabel("NDCG@10")
    left.set_ylim(bottom=0)

    base_comparison = (
        paired.loc[
            (paired["metric"] == "ndcg")
            & (paired["model"] == "C: Cognitive")
            & (paired["compared_with"] == "B3: EASE")
        ]
        .set_index("window")
        .loc[windows["window"]]
    )
    right.axhline(0, color=INK_MUTED, linewidth=1, linestyle="--")
    _interval(
        right,
        x,
        base_comparison["difference"],
        base_comparison["ci_low"],
        base_comparison["ci_high"],
    )
    _title(right, "Cognitive minus EASE, base scores")
    right.set_ylabel("Paired NDCG@10 difference")
    for axis in (left, right):
        axis.set_xticks(x, labels)
        axis.set_xlim(-0.25, len(x) - 0.75)
    _legend(figure, *left.get_legend_handles_labels(), columns=4)
    return _save(figure, path)


def _policy_effects(effects: pd.DataFrame, windows: pd.DataFrame, path: Path) -> Path:
    figure, axes = _new_figure(8.6, 5.25, columns=2, rows=2)
    specs = (
        ("ndcg", 1, "Ranking quality", "Δ NDCG@10"),
        ("coverage_pct", 1, "Catalogue coverage", "Δ coverage (percentage points)"),
        ("rank_merit_gap", 1, "Exposure alignment", "Δ rank-weighted merit gap"),
        ("recall_tail", 100, "Tail retrieval", "Δ tail recall (percentage points)"),
    )
    x = np.arange(len(windows))
    for axis, (column, scale, title, unit) in zip(axes, specs, strict=True):
        axis.axhline(0, color=INK_MUTED, linewidth=1, linestyle="--")
        for offset, (scorer, (color, marker)) in zip(
            (-0.10, 0.10), SCORER_STYLES.items(), strict=True
        ):
            rows = (
                effects.loc[effects["scorer"] == scorer].set_index("window").loc[windows["window"]]
            )
            position = x + offset
            if column == "ndcg":
                axis.vlines(position, rows["ndcg_ci_low"], rows["ndcg_ci_high"], color=color)
            axis.scatter(
                position,
                scale * rows[column],
                color=color,
                marker=marker,
                edgecolor=SURFACE,
                linewidth=0.7,
                s=43,
                zorder=3,
                label=scorer,
            )
        _title(axis, title)
        axis.set_xticks(x, windows["window"].str.title())
        axis.set_xlim(-0.35, len(x) - 0.65)
        axis.set_ylabel(unit)
    _legend(figure, *axes[0].get_legend_handles_labels())
    return _save(figure, path)


def _validation_budget(grid: pd.DataFrame, budget_pct: float, path: Path) -> Path:
    windows = list(grid["window"].unique())
    figure, axes = _new_figure(10, 3.55, columns=len(windows))
    fitted = grid.loc[grid["ndcg_loss_pct"] <= budget_pct + 5]
    left_limit = min(-1.0, float(fitted["ndcg_loss_pct"].min()) - 1)
    coverage = fitted["coverage_pct"]
    coverage_margin = max(0.5, 0.06 * (coverage.max() - coverage.min()))
    for axis, window in zip(axes, windows, strict=True):
        axis.axvline(budget_pct, color=INK_MUTED, linewidth=1, linestyle="--")
        for scorer, (color, marker) in SCORER_STYLES.items():
            rows = fitted.loc[(fitted["scorer"] == scorer) & (fitted["window"] == window)]
            axis.scatter(
                rows["ndcg_loss_pct"],
                rows["coverage_pct"],
                color=color,
                marker=marker,
                s=26,
                alpha=0.72,
                label=scorer,
            )
            selected = rows.loc[rows["selected"]]
            axis.scatter(
                selected["ndcg_loss_pct"],
                selected["coverage_pct"],
                color=color,
                marker="D",
                s=65,
                edgecolor="#0b0b0b",
                linewidth=0.9,
                zorder=4,
            )
        _title(axis, f"{window.title()} validation window")
        axis.set_xlim(left_limit, budget_pct + 5.5)
        axis.set_ylim(coverage.min() - coverage_margin, coverage.max() + coverage_margin)
        axis.set_xlabel("Relative NDCG@10 loss (%)")
    axes[0].set_ylabel("Catalogue coverage (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.extend(
        [
            Line2D(
                [], [], marker="D", color="white", markeredgecolor="black", markerfacecolor="gray"
            ),
            Line2D([], [], color=INK_MUTED, linestyle="--"),
        ]
    )
    _legend(figure, handles, [*labels, "Selected", f"{budget_pct:g}% loss budget"], columns=4)
    return _save(figure, path)


def _ablations(comparisons: pd.DataFrame, path: Path) -> Path:
    figure, axes = _new_figure(8.7, 3.5, columns=2)
    ablations = {
        "w/o short-term memory": "Remove recent memory",
        "uniform attention": "Uniform attention",
        "w/o RPE adaptation": "Remove reward-error gate",
        "w/o exploration (Model C)": "Remove exploration",
    }
    selected = comparisons.loc[
        (comparisons["compared_with"] == "Full Model D")
        & comparisons["metric"].isin(["ndcg", "replay_ndcg"])
        & comparisons["model"].isin(ablations)
    ]
    span = selected["ci_high"].max() - selected["ci_low"].min()
    margin = max(span * 0.05, 0.001)
    for axis, column, title in zip(
        axes, ("ndcg", "replay_ndcg"), ("One-shot ranking", "Logged replay"), strict=True
    ):
        rows = selected.loc[selected["metric"] == column].set_index("model").loc[list(ablations)]
        y = np.arange(len(ablations))
        axis.axvline(0, color=INK_MUTED, linewidth=1, linestyle="--")
        _interval(axis, rows["difference"], y, rows["ci_low"], rows["ci_high"], horizontal=True)
        _title(axis, title)
        axis.set_yticks(y, list(ablations.values()) if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_xlabel("NDCG@10 change relative to full D")
        axis.set_xlim(selected["ci_low"].min() - margin, selected["ci_high"].max() + margin)
    return _save(figure, path)


def _fidelity(metrics: pd.DataFrame, path: Path) -> Path:
    figure, (gap, tau) = _new_figure(8.65, 3.3, columns=2)
    rows = metrics.loc[metrics["fidelity_users"] > 0]
    y = np.arange(len(rows))
    gap.axvline(0, color=INK_MUTED, linewidth=1, linestyle="--")
    tau.axvline(0, color=INK_MUTED, linewidth=1, linestyle="--")
    _interval(
        gap,
        rows["fidelity_gap"],
        y,
        rows["fidelity_gap_low"],
        rows["fidelity_gap_high"],
        horizontal=True,
    )
    tau.scatter(rows["kendall_tau"], y, color=SERIES_COLOR, s=32, zorder=3)
    _title(gap, "Highest-weight evidence deletion")
    _title(tau, "Evidence ordering agreement")
    gap.set_xlabel("Standardised deletion gap")
    tau.set_xlabel("Mean Kendall tau-b")
    tau.set_xlim(-1.05, 1.05)
    gap.set_yticks(y, rows["model"].str.split(": ", n=1).str[-1])
    tau.set_yticks(y, [])
    for axis in (gap, tau):
        axis.set_ylim(len(rows) - 0.5, -0.5)
    return _save(figure, path)


def _stress(metrics: pd.DataFrame, path: Path) -> Path:
    models = {
        "B3: EASE": ("EASE", "#00785d", "s"),
        "B4: All-history profile": ("All-history profile", "#5b6573", "^"),
        "B5: Recency profile": ("Recency profile", "#a24510", "v"),
        "C: Cognitive": ("Cognitive", "#1565a0", "o"),
        "D: Cognitive + UCB": ("Cognitive + exploration", "#9c3d77", "D"),
    }
    figure, (left, right) = _new_figure(8.8, 3.85, columns=2)
    rows = metrics.set_index("model").loc[list(models)]
    for model, (label, color, marker) in models.items():
        row = rows.loc[model]
        for axis, column in ((left, "shift_retention_pct"), (right, "noise_retention_5_pct")):
            axis.scatter(
                100 * row["shift_feedback_gain"],
                row[column],
                color=color,
                marker=marker,
                edgecolor=SURFACE,
                linewidth=0.8,
                s=67,
                label=label,
            )
    _title(left, "Shift responsiveness and retention")
    _title(right, "Shift responsiveness and noise robustness")
    left.set_ylabel("Original NDCG retained after shift (%)")
    right.set_ylabel("NDCG retained after random ratings (%)")
    retention = rows[["shift_retention_pct", "noise_retention_5_pct"]].max().max()
    for axis in (left, right):
        axis.axhline(100, color=INK_MUTED, linewidth=1, linestyle="--")
        axis.set_xlabel("Genre exposure gain after shift (pp)")
        axis.set_ylim(0, max(108, retention + 5))
        axis.set_xlim(left=min(0, 100 * rows["shift_feedback_gain"].min() - 0.3))
    _legend(figure, *left.get_legend_handles_labels(), columns=5)
    return _save(figure, path)


def render(
    exploration: Path,
    rolling: Path,
    target: Path,
    *,
    trust: Path | None = None,
    ablation: Path | None = None,
) -> list[Path]:
    """Render the final figure set, or its follow-up subset when core runs are unavailable."""
    sources = {}

    def read(directory: Path, filename: str) -> Path:
        source = directory / filename
        sources.setdefault(str(source), hashlib.sha256(source.read_bytes()).hexdigest())
        return source

    windows = pd.read_csv(read(exploration, "windows.csv"))
    exploration_config = yaml.safe_load(
        read(exploration, "config.yaml").read_text(encoding="utf-8")
    )
    budget = 100 * exploration_config["config"]["exploration_study"]["max_ndcg_loss"]
    figures = [
        _ranking(
            pd.read_csv(read(exploration, "metrics.csv")),
            pd.read_csv(read(rolling, "comparisons.csv")),
            windows,
            target / "01_rolling_ranking.png",
        ),
        _policy_effects(
            pd.read_csv(read(exploration, "effects.csv")),
            windows,
            target / "02_exploration_effects.png",
        ),
        _validation_budget(
            pd.read_csv(read(exploration, "validation.csv")),
            budget,
            target / "03_validation_budget.png",
        ),
        _stress(
            pd.read_csv(read(rolling, "anchor_reliability.csv")),
            target / "07_reliability_stress.png",
        ),
    ]
    if trust is not None:
        figures.extend(
            [
                _fidelity(
                    pd.read_csv(read(trust, "metrics.csv")), target / "05_explanation_fidelity.png"
                ),
                plot_shift_curves(
                    pd.read_csv(read(trust, "shift_curves.csv")), target / "06_preference_shift.png"
                ),
            ]
        )
    if ablation is not None:
        models = yaml.safe_load(read(ablation, "models.yaml").read_text(encoding="utf-8"))
        config = yaml.safe_load(read(ablation, "config.yaml").read_text(encoding="utf-8"))
        figures.extend(
            [
                _ablations(
                    pd.read_csv(read(ablation, "comparisons.csv")), target / "04_ablations.png"
                ),
                plot_adaptation_stability(
                    pd.read_csv(read(ablation, "alpha_sweep.csv")),
                    models["Full Model D"]["params"]["base_alpha"],
                    config["config"]["reliability"]["shift_report_step"],
                    target / "08_plasticity_tradeoff.png",
                ),
            ]
        )
    renderer_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(plots.__file__))
    }
    with (target / "sources.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "inputs_sha256": sources,
                "renderer_sha256": renderer_hashes,
                "png_dpi": 300,
                "svg_text": "outlined",
            },
            handle,
        )
    return sorted(figures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exploration", type=Path, required=True)
    parser.add_argument("--rolling", type=Path, required=True)
    parser.add_argument("--trust", type=Path)
    parser.add_argument("--ablation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = args.output or args.exploration / "figures"
    figures = render(
        args.exploration, args.rolling, target, trust=args.trust, ablation=args.ablation
    )
    print(f"{len(figures)} figures exported as SVG and PNG to {target}")


if __name__ == "__main__":
    main()
