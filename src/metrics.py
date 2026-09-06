"""Shared evaluation metrics for every model in the project."""

import numpy as np
from sklearn.metrics import confusion_matrix

# Order matters: index 0 = human, index 1 = AI, matching the label
# encoding set in data_prep.py and relied on by explain.py.
LABELS = [0, 1]


def evaluate(y_true, y_pred):
    """Return the four metrics as a dict of floats in [0, 1]."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / cm.sum() if cm.sum() else 0.0

    # Precision/recall of the AI class (label 1), then their harmonic mean.
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity)
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def confusion(y_true, y_pred):
    """Raw 2x2 confusion matrix, rows = true, cols = predicted, [human, ai]."""
    return confusion_matrix(y_true, y_pred, labels=LABELS)


def bootstrap_ci(y_true, y_pred, metric="accuracy", n_resamples=1000, seed=42, alpha=0.05):
    """Bootstrap a (lo, point, hi) triple for one metric."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    point = evaluate(y_true, y_pred)[metric]

    scores = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores[i] = evaluate(y_true[idx], y_pred[idx])[metric]

    lo = float(np.percentile(scores, 100 * alpha / 2))
    hi = float(np.percentile(scores, 100 * (1 - alpha / 2)))

    # Clamp so the point estimate always sits inside the reported interval.
    return (min(lo, point), point, max(hi, point))


def format_table(rows, headers):
    """Render a list of row-lists as a plain-text table."""
    cols = list(zip(*([headers] + rows))) if rows else [headers]
    widths = [max(len(str(cell)) for cell in col) for col in cols]

    def line(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out.extend(line(r) for r in rows)
    return "\n".join(out)
