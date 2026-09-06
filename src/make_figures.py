"""Generates the charts used in docs/REPORT.md from
results/decision_matrix.json."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import DECISION_MATRIX, FIGURES_DIR, ensure_dirs


def load():
    with open(DECISION_MATRIX) as fh:
        return json.load(fh)


def plot_confusion_matrices(payload, path):
    models = payload["models"]
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.8))
    for ax, (name, rec) in zip(np.atleast_1d(axes), models.items()):
        cm = np.array(rec["confusion"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                # Flip the label colour on dark cells so it stays readable.
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=13,
                        color="white" if cm[i, j] > cm.max() * 0.6 else "black")
        ax.set_xticks([0, 1], ["human", "ai"])
        ax.set_yticks([0, 1], ["human", "ai"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name}\nacc {rec['metrics']['accuracy']:.4f}", fontsize=10)
    fig.suptitle("Confusion matrices (600-row test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_criteria(payload, path):
    """The five criteria side by side."""
    names = list(payload["models"])
    criteria = payload["criteria"]
    fig, axes = plt.subplots(1, 5, figsize=(19, 4))

    for ax, crit in zip(axes, criteria):
        values = [payload["models"][n]["crisp"][crit] for n in names]
        is_cost = payload["directions"][criteria.index(crit)] == "cost"
        ax.bar(range(len(names)), values,
               color=["#888"] * (len(names) - 1) + ["#c44"])
        if is_cost:
            ax.set_yscale("log")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        spread = max(values) / min(values) if min(values) > 0 else float("inf")
        arrow = "lower is better" if is_cost else "higher is better"
        ax.set_title(f"{crit}\n{arrow} — spread {spread:.1f}x", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Decision criteria. Cost criteria are log-scaled: they span up to "
        "1168x across models, quality only 1.1x"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_quality_intervals(payload, path):
    """Accuracy with 95% bootstrap intervals, which show which gaps between
    models are real and which are within sampling noise."""
    names = list(payload["models"])
    points = [payload["models"][n]["crisp"]["accuracy"] for n in names]
    tfns = [payload["models"][n]["intervals"]["accuracy"] for n in names]
    lows = [p - t[0] for p, t in zip(points, tfns)]
    highs = [t[2] - p for p, t in zip(points, tfns)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(range(len(names)), points, yerr=[lows, highs],
                fmt="o", capsize=6, markersize=8, linewidth=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy with 95% bootstrap confidence intervals\n"
                 "(overlapping intervals = difference is not significant)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_linguistic_matrix(payload, path):
    """The linguistic ratings, as a categorical grid."""
    names = list(payload["models"])
    criteria = payload["criteria"]
    order = ["poor", "fair", "good", "very good", "excellent", "outstanding"]
    rank = {c: i for i, c in enumerate(order)}

    grid = np.array([[rank[payload["models"][n]["linguistic"][c]["category"]]
                      for c in criteria] for n in names], dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    im = ax.imshow(grid, cmap="YlGnBu", vmin=0, vmax=len(order) - 1, aspect="auto")

    for i in range(len(names)):
        for j in range(len(criteria)):
            label = payload["models"][names[i]]["linguistic"][criteria[j]]["category"]
            ax.text(j, i, label, ha="center", va="center", fontsize=8.5,
                    color="white" if grid[i, j] > 3 else "black")

    ax.set_xticks(range(len(criteria)))
    ax.set_xticklabels(criteria, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title("Linguistic ratings — the decision matrix for fuzzy TOPSIS")

    cbar = fig.colorbar(im, ax=ax, ticks=range(len(order)), pad=0.02)
    cbar.ax.set_yticklabels(order, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ensure_dirs()
    payload = load()

    for fn, name in (
        (plot_confusion_matrices, "confusion_matrices.png"),
        (plot_criteria, "criteria_comparison.png"),
        (plot_quality_intervals, "quality_intervals.png"),
        (plot_linguistic_matrix, "linguistic_matrix.png"),
    ):
        path = FIGURES_DIR / name
        fn(payload, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
