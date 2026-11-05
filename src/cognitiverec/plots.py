"""Report figures. Owner: Abhinav.

Colours are fixed per model and were checked with a colourblind-safety validator. Three of them
have low contrast on the background, so each model also has its own marker shape, and every figure
sits next to a CSV of the same numbers. Each plot has one y-axis: quantities on different scales
get separate panels rather than a second axis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Colour follows the model, in this order, in every figure.
MODEL_STYLES = {
    "A: Popularity": ("#2a78d6", "o"),
    "B1: ItemKNN": ("#eb6834", "s"),
    "B2: SVD": ("#1baf7a", "^"),
    "C: Cognitive": ("#eda100", "D"),
    "D: Cognitive + UCB": ("#e87ba4", "v"),
}
SERIES_COLOR = "#2a78d6"

LINE_WIDTH = 1.5  # points; 2px at 96 dpi
MARKER_SIZE = 7
RING_WIDTH = 1.5


def _new_figure(width: float, height: float, columns: int = 1) -> tuple[Figure, np.ndarray]:
    figure = Figure(figsize=(width, height), dpi=200, facecolor=SURFACE, layout="constrained")
    axes = np.atleast_1d(figure.subplots(1, columns))
    for ax in axes:
        _style_axes(ax)
    return figure, axes


def _style_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.75, linestyle="solid")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(length=0, labelcolor=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.xaxis.label.set_size(9.5)
    ax.yaxis.label.set_size(9.5)


def _title(ax: Axes, text: str) -> None:
    ax.set_title(text, loc="left", color=INK, fontsize=11, fontweight="bold", pad=10)


def _line(ax: Axes, x, y, color: str, marker: str, label: str | None = None) -> None:
    ax.plot(
        x,
        y,
        color=color,
        linewidth=LINE_WIDTH,
        marker=marker,
        markersize=MARKER_SIZE,
        markeredgecolor=SURFACE,
        markeredgewidth=RING_WIDTH,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=label,
    )


def _note(
    ax: Axes, text: str, x: float, y: float, offset: tuple[int, int], ha: str = "left"
) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=8.5,
        color=INK_SECONDARY,
    )


def _save(figure: Figure, path: Path) -> Path:
    figure.savefig(path, facecolor=SURFACE)
    return path


def plot_shift_curves(curves: pd.DataFrame, path: Path) -> Path:
    """Two panels: how fast each model adapts to a new genre, and what it forgets on the way."""
    figure, (left, right) = _new_figure(9.0, 3.8, columns=2)
    for model, group in curves.groupby("model", sort=False):
        color, marker = MODEL_STYLES.get(model, (INK_MUTED, "o"))
        _line(left, group["step"], 100 * group["adaptation"], color, marker, model)
        _line(right, group["step"], group["retention_pct"], color, marker, model)

    _title(left, "Adaptation to a new genre")
    left.set_xlabel("Interactions injected from the new genre")
    left.set_ylabel("Share of top-k in the new genre (%)")
    left.set_ylim(bottom=0)
    _title(right, "Retention of existing interests")
    right.set_xlabel("Interactions injected from the new genre")
    right.set_ylabel("NDCG on original positives (% of start)")

    handles, labels = left.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=len(labels),
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_SECONDARY,
    )
    return _save(figure, path)


def plot_accuracy_coverage(
    sweep: pd.DataFrame, baselines: pd.DataFrame, selected_beta: float, path: Path
) -> Path:
    """NDCG against catalogue coverage as the exploration weight grows, on one axis."""
    figure, axes = _new_figure(6.4, 4.2)
    ax = axes[0]
    _line(ax, sweep["coverage_pct"], sweep["ndcg"], SERIES_COLOR, "o", "Model D as beta grows")

    ax.scatter(
        baselines["coverage_pct"],
        baselines["ndcg"],
        s=MARKER_SIZE**2,
        color=INK_MUTED,
        edgecolor=SURFACE,
        linewidth=RING_WIDTH,
        zorder=3,
        label="Tuned baselines",
    )
    for row in baselines.itertuples():
        _note(ax, row.model, row.coverage_pct, row.ndcg, (8, 0))

    first, last = sweep.iloc[0], sweep.iloc[-1]
    _note(ax, f"beta = {first.exploration_beta:g}", first.coverage_pct, first.ndcg, (8, 8))
    _note(
        ax, f"beta = {last.exploration_beta:g}", last.coverage_pct, last.ndcg, (-8, 10), ha="right"
    )
    selected = sweep[np.isclose(sweep["exploration_beta"], selected_beta)]
    if len(selected) and selected_beta not in (first.exploration_beta, last.exploration_beta):
        row = selected.iloc[0]
        _note(ax, f"selected, beta = {selected_beta:g}", row.coverage_pct, row.ndcg, (8, 10))

    _title(ax, "Accuracy against catalogue coverage (validation)")
    ax.set_xlabel("Catalogue coverage (%)")
    ax.set_ylabel("NDCG@10")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, loc="upper right")
    return _save(figure, path)


def plot_adaptation_stability(
    sweep: pd.DataFrame, selected_alpha: float, report_step: int, path: Path
) -> Path:
    """Adaptation against list stability as base plasticity grows, on one axis."""
    figure, axes = _new_figure(6.4, 4.2)
    ax = axes[0]
    _line(ax, sweep["stability"], 100 * sweep["shift_adaptation"], SERIES_COLOR, "o")

    first, last = sweep.iloc[0], sweep.iloc[-1]
    _note(
        ax,
        f"alpha = {first.base_alpha:g}",
        first.stability,
        100 * first.shift_adaptation,
        (-8, -10),
        ha="right",
    )
    _note(ax, f"alpha = {last.base_alpha:g}", last.stability, 100 * last.shift_adaptation, (8, 10))
    selected = sweep[np.isclose(sweep["base_alpha"], selected_alpha)]
    if len(selected) and selected_alpha not in (first.base_alpha, last.base_alpha):
        row = selected.iloc[0]
        _note(
            ax,
            f"selected, alpha = {selected_alpha:g}",
            row.stability,
            100 * row.shift_adaptation,
            (8, 10),
        )

    ax.margins(x=0.08, y=0.14)  # room for the end labels inside the axes
    _title(ax, "Adaptation against stability as plasticity grows (validation)")
    ax.set_xlabel("Stability: overlap between consecutive top-k lists (Jaccard)")
    ax.set_ylabel(f"Top-k share of the new genre after {report_step} injections (%)")
    return _save(figure, path)
