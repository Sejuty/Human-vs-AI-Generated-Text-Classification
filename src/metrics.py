"""
metrics.py
----------
Shared evaluation metrics for every model in the project, so that
train.py, train_candidates.py, train_advanced.py and benchmark.py all
score models the same way instead of each re-deriving its own numbers.

Four metrics, all read off the 2x2 confusion matrix (0 = human, 1 = AI):

    Accuracy    = (TP + TN) / total          — overall correctness
    F1          = harmonic mean of precision and recall for the AI class
    Sensitivity = TP / (TP + FN)             — recall of the AI class
    Specificity = TN / (TN + FP)             — recall of the human class

Sensitivity and Specificity split "accuracy" into its two failure
directions: missing AI text vs. falsely accusing a human. On a balanced
test set those two average out to exactly the accuracy, which is worth
knowing — it is why these four metrics alone cannot serve as
independent criteria for the multi-criteria ranking in topsis.py.

Why bootstrap confidence intervals
----------------------------------
Every metric here is a point estimate computed from 600 test rows, so it
carries sampling noise: measure the same model on a different 600 rows
and the number moves. `bootstrap_ci` quantifies that by resampling the
test set with replacement many times and re-scoring each resample. The
2.5th/97.5th percentiles of the resulting spread give a 95% interval.

The intervals answer a question the point estimates cannot: which gaps
between models are real. Two models whose intervals overlap heavily are
not meaningfully different on this test set, however their headline
accuracies are ordered.
"""

import numpy as np
from sklearn.metrics import confusion_matrix

# Order matters: index 0 = human, index 1 = AI, matching the label
# encoding set in data_prep.py and relied on by explain.py.
LABELS = [0, 1]


def evaluate(y_true, y_pred):
    """
    Return the four metrics as a dict of floats in [0, 1].

    Computed from the confusion matrix directly rather than via
    sklearn's individual scorers, so that all four stay consistent with
    the same TN/FP/FN/TP counts, and so `labels=LABELS` pins the matrix
    orientation even if a split happened to contain only one class.
    """
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / cm.sum() if cm.sum() else 0.0

    # Precision/recall of the AI class (label 1), then their harmonic
    # mean. Guarded because a model that never predicts AI has an
    # undefined precision.
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
    """
    Bootstrap a (lo, point, hi) triple for one metric.

    Resamples *indices* of the held-out predictions with replacement, so
    the model is never re-run — we are quantifying uncertainty in the
    test sample, not in the training procedure. This makes it cheap
    enough to run for every model and metric.

    `point` is the metric on the full test set rather than the bootstrap
    mean, so it stays consistent with the headline number printed by the
    training scripts.
    """
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

    # Clamp so the point estimate always sits inside the reported
    # interval; with a skewed bootstrap distribution it can otherwise
    # fall marginally outside, which reads as an error to anyone
    # scanning the table.
    return (min(lo, point), point, max(hi, point))


def format_table(rows, headers):
    """
    Render a list of row-lists as a plain-text table.

    Used by benchmark.py and select_model.py so their intermediate
    tables (decision matrix, normalised matrix, distances) all print in
    the same shape without pulling in a dependency just for formatting.
    """
    cols = list(zip(*([headers] + rows))) if rows else [headers]
    widths = [max(len(str(cell)) for cell in col) for col in cols]

    def line(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out.extend(line(r) for r in rows)
    return "\n".join(out)
