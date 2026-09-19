"""Chart helpers for the churn project.

One palette, one style, applied everywhere, so every figure in the report reads
as part of the same system. Colours come from a CVD-validated categorical set;
hues are assigned in fixed slot order and never cycled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from churn_lib import FIGURE_DIR

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
SURFACE: Final[str] = "#fcfcfb"
INK: Final[str] = "#0b0b0b"
INK_SECONDARY: Final[str] = "#52514e"
INK_MUTED: Final[str] = "#84837c"
GRID: Final[str] = "#e6e5e1"
NEUTRAL: Final[str] = "#c7c5be"

#: Categorical slots, in fixed order. Slot 1 = retained, slot 2 = churned.
SERIES: Final[list[str]] = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
RETAINED: Final[str] = SERIES[0]
CHURNED: Final[str] = SERIES[1]

#: Single-hue blue ramp, light -> dark, for magnitude encoding.
SEQ_BLUE: Final[list[str]] = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


def apply_style() -> None:
    """Install the project-wide matplotlib defaults: recessive chrome, ink text."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.dpi": 150,
            "figure.dpi": 110,
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.0,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlepad": 12,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.9,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _finish(ax: Axes, *, xgrid: bool = False) -> None:
    """Show grid lines on one axis only, so the chrome stays recessive."""
    ax.grid(axis="x" if xgrid else "y", visible=True)
    ax.grid(axis="y" if xgrid else "x", visible=False)


def save(fig: Figure, name: str, directory: Path = FIGURE_DIR) -> Path:
    """Write a figure to the reports folder and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    return path


def _ramp(values: Sequence[float], *, lo: int = 3, hi: int = 11) -> list[str]:
    """Map magnitudes onto the sequential blue ramp (larger value -> darker)."""
    arr = np.asarray(values, dtype=float)
    span = arr.max() - arr.min()
    norm = np.zeros_like(arr) if span == 0 else (arr - arr.min()) / span
    idx = np.round(lo + norm * (hi - lo)).astype(int)
    return [SEQ_BLUE[i] for i in idx]


# --------------------------------------------------------------------------- #
# EDA charts
# --------------------------------------------------------------------------- #
def plot_target_balance(y: pd.Series) -> Figure:
    """Part-to-whole bar of retained vs. churned customers, directly labelled."""
    counts = y.value_counts().reindex([0, 1]).fillna(0).astype(int)
    total = int(counts.sum())
    churn_rate = counts[1] / total

    fig, ax = plt.subplots(figsize=(8, 2.1))
    left = 0.0
    for value, colour, label in ((0, RETAINED, "Retained"), (1, CHURNED, "Churned")):
        width = counts[value] / total
        ax.barh(
            0, width, left=left, height=0.5, color=colour, label=label,
            edgecolor=SURFACE, linewidth=2.0,
        )
        ax.text(
            left + width / 2, 0, f"{label}\n{counts[value]:,}  ({width:.1%})",
            ha="center", va="center", color=SURFACE, fontsize=10, fontweight="bold",
        )
        left += width

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.45, 0.45)
    ax.axis("off")
    ax.set_title(
        f"Class balance: {churn_rate:.1%} of {total:,} customers churned",
        loc="left", pad=14,
    )
    fig.text(
        0.0, -0.10,
        f"Imbalance ratio {counts[0] / counts[1]:.1f} : 1 -- accuracy is not a usable metric here.",
        color=INK_MUTED, fontsize=9,
    )
    return fig


def plot_churn_rate_by_category(
    df: pd.DataFrame, column: str, target: str = "churn_flag", *, title: str | None = None
) -> Figure:
    """Horizontal bars of churn rate per level, shaded by magnitude.

    Args:
        df: Cleaned frame carrying ``column`` and a binary ``target``.
        column: Categorical column to segment by.
        target: Name of the 0/1 churn column.
        title: Optional override for the chart title.
    """
    grouped = (
        df.groupby(column, observed=True)[target]
        .agg(rate="mean", n="size")
        .sort_values("rate")
    )
    colours = _ramp(grouped["rate"].to_numpy())
    baseline = df[target].mean()

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(grouped) + 2.0))
    ax.barh(grouped.index, grouped["rate"], color=colours, height=0.62,
            edgecolor=SURFACE, linewidth=2.0)
    ax.axvline(baseline, color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=3)
    # Anchor the reference label inside the axes, just above the bottom spine.
    ax.text(baseline, -0.45, f" overall {baseline:.0%}",
            color=INK_MUTED, fontsize=9, va="bottom", ha="left")

    for y_pos, (rate, n) in enumerate(zip(grouped["rate"], grouped["n"])):
        ax.text(rate + 0.012, y_pos, f"{rate:.0%}   n={n:,}",
                va="center", color=INK_SECONDARY, fontsize=9)

    ax.set_xlim(0, min(1.0, grouped["rate"].max() * 1.35))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Churn rate")
    ax.set_title(title or f"Churn rate by {column}", loc="left")
    _finish(ax, xgrid=True)
    return fig


def plot_distribution_by_churn(
    df: pd.DataFrame, column: str, target: str = "churn_flag", *,
    bins: int = 30, title: str | None = None, xlabel: str | None = None,
) -> Figure:
    """Overlaid histograms of a numeric column, split by churn outcome."""
    retained = df.loc[df[target] == 0, column].dropna()
    churned = df.loc[df[target] == 1, column].dropna()
    edges = np.histogram_bin_edges(df[column].dropna(), bins=bins)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for data, colour, label in ((retained, RETAINED, "Retained"), (churned, CHURNED, "Churned")):
        ax.hist(data, bins=edges, density=True, color=colour, alpha=0.55,
                label=f"{label} (n={len(data):,})")
        ax.hist(data, bins=edges, density=True, histtype="step", color=colour, linewidth=2.0)

    ax.set_xlabel(xlabel or column)
    ax.set_ylabel("Share of group")
    # Density values are not meaningful to a reader; the shape is the message.
    ax.set_yticks([])
    ax.set_title(title or f"{column} distribution by outcome", loc="left")
    ax.legend(loc="upper right")
    _finish(ax)
    return fig


def plot_churn_curve(
    df: pd.DataFrame, column: str, target: str = "churn_flag", *,
    title: str | None = None, xlabel: str | None = None,
) -> Figure:
    """Line of churn rate across an ordered segment, with the overall rate marked."""
    grouped = df.groupby(column, observed=True)[target].agg(rate="mean", n="size")
    baseline = df[target].mean()

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(range(len(grouped)), grouped["rate"], color=SERIES[0], marker="o",
            markersize=9, markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=4)
    ax.axhline(baseline, color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)))
    ax.text(len(grouped) - 0.6, baseline + 0.012, f"overall {baseline:.0%}",
            color=INK_MUTED, fontsize=9)

    for x_pos, rate in enumerate(grouped["rate"]):
        ax.annotate(f"{rate:.0%}", (x_pos, rate), textcoords="offset points",
                    xytext=(0, 12), ha="center", color=INK_SECONDARY, fontsize=9)

    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped.index)
    ax.set_ylim(0, grouped["rate"].max() * 1.28)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.set_xlabel(xlabel or column)
    ax.set_ylabel("Churn rate")
    ax.set_title(title or f"Churn rate across {column}", loc="left")
    _finish(ax)
    return fig


# --------------------------------------------------------------------------- #
# Model-evaluation charts
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(cm: np.ndarray, *, title: str = "Confusion matrix") -> Figure:
    """Sequential-blue confusion matrix annotated with counts and row shares."""
    labels = ["Retained", "Churned"]
    row_totals = cm.sum(axis=1, keepdims=True)
    shares = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(shares, cmap=mpl.colors.LinearSegmentedColormap.from_list("seq", SEQ_BLUE),
              vmin=0, vmax=1)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}\n{shares[i, j]:.1%}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color=SURFACE if shares[i, j] > 0.5 else INK)

    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, loc="left")
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.5)
    ax.tick_params(which="minor", length=0)
    return fig


def plot_roc_pr(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    *, kind: str = "roc", positive_rate: float | None = None,
) -> Figure:
    """Plot ROC or precision-recall curves for one or more models.

    Args:
        curves: ``{model_name: (x, y, auc_score)}``.
        kind: ``"roc"`` or ``"pr"``.
        positive_rate: Base churn rate, drawn as the PR no-skill line.
    """
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for slot, (name, (x, y, score)) in enumerate(curves.items()):
        ax.plot(x, y, color=SERIES[slot % len(SERIES)], label=f"{name}  ({score:.3f})")

    if kind == "roc":
        ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=1.4, linestyle=(0, (4, 3)),
                label="No skill (0.500)")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate (recall)")
        ax.set_title("ROC curves -- test set", loc="left")
    else:
        if positive_rate is not None:
            ax.axhline(positive_rate, color=NEUTRAL, linewidth=1.4, linestyle=(0, (4, 3)),
                       label=f"No skill ({positive_rate:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-recall curves -- test set", loc="left")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right" if kind == "roc" else "upper right")
    _finish(ax)
    ax.grid(axis="x", visible=True)
    return fig


def plot_threshold_sweep(
    thresholds: np.ndarray, metrics: dict[str, np.ndarray], *, chosen: float | None = None
) -> Figure:
    """Precision / recall / F1 against the decision threshold, on one shared axis."""
    # Fixed, well-separated label anchors keep direct labels off each other and
    # off the figure edge -- peaks would bunch the three curves together.
    anchors = (0.16, 0.40, 0.64, 0.84)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for slot, (name, values) in enumerate(metrics.items()):
        colour = SERIES[slot % len(SERIES)]
        ax.plot(thresholds, values, color=colour, label=name)
        at = int(np.argmin(np.abs(thresholds - anchors[slot % len(anchors)])))
        ax.annotate(name, (thresholds[at], values[at]), textcoords="offset points",
                    xytext=(0, 9), ha="center", color=INK_SECONDARY, fontsize=9)

    if chosen is not None:
        ax.axvline(chosen, color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)))
        # Top of the plot: the curves have all descended by the time they cross it.
        ax.text(chosen + 0.012, 1.03, f"chosen {chosen:.3f}",
                color=INK_MUTED, fontsize=9, ha="left", va="top")

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.06)
    ax.set_title("Operating point trade-off", loc="left")
    # Lower-left is the one region all three curves leave empty.
    ax.legend(loc="lower left", ncol=1)
    _finish(ax)
    return fig


def plot_feature_importance(
    names: Sequence[str], values: Sequence[float], *,
    top_n: int = 15, title: str = "Feature importance", xlabel: str = "Importance",
    signed: bool = False,
) -> Figure:
    """Ranked horizontal bars of feature importance.

    Args:
        names: Feature labels.
        values: Importance scores, aligned with ``names``.
        top_n: How many features to show.
        title: Chart title.
        xlabel: Axis label for the score.
        signed: When ``True``, colour by direction (churn-increasing vs. -decreasing)
            instead of by magnitude -- used for logistic-regression coefficients.
    """
    frame = (
        pd.DataFrame({"feature": list(names), "value": list(values)})
        .assign(magnitude=lambda d: d["value"].abs())
        .nlargest(top_n, "magnitude")
        .sort_values("magnitude")
    )

    if signed:
        colours = [CHURNED if v > 0 else RETAINED for v in frame["value"]]
    else:
        colours = _ramp(frame["magnitude"].to_numpy())

    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(frame) + 1.9))
    ax.barh(frame["feature"], frame["value"] if signed else frame["magnitude"],
            color=colours, height=0.66, edgecolor=SURFACE, linewidth=2.0)

    if signed:
        ax.axvline(0, color=INK_SECONDARY, linewidth=1.2)
        handles = [
            mpl.patches.Patch(color=CHURNED, label="Raises churn risk"),
            mpl.patches.Patch(color=RETAINED, label="Lowers churn risk"),
        ]
        ax.legend(handles=handles, loc="lower right")

    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    _finish(ax, xgrid=True)
    return fig


def plot_model_comparison(scores: pd.DataFrame, *, metric: str = "roc_auc") -> Figure:
    """Ranked bars comparing candidate models on a single metric."""
    ranked = scores.sort_values(metric)
    colours = _ramp(ranked[metric].to_numpy())

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(ranked) + 2.0))
    ax.barh(ranked.index, ranked[metric], color=colours, height=0.6,
            edgecolor=SURFACE, linewidth=2.0)
    for y_pos, value in enumerate(ranked[metric]):
        ax.text(value + 0.005, y_pos, f"{value:.3f}", va="center",
                color=INK_SECONDARY, fontsize=9)

    ax.set_xlim(0, min(1.0, ranked[metric].max() * 1.15))
    ax.set_xlabel(metric.replace("_", " ").upper())
    ax.set_title(f"Model comparison -- {metric.replace('_', ' ').upper()}", loc="left")
    _finish(ax, xgrid=True)
    return fig
