"""
benchmark.py
------------
Measures every criterion for every candidate model and writes the
decision matrix that select_model.py ranks.

This is the only place measurements are taken. Keeping them in one file
means the ranking can be re-run with different weights without re-timing
anything, and guarantees all four models are measured under identical
conditions — which matters, because timings compared across different
runs or machines would be meaningless.

The five criteria
-----------------
Two are quality criteria, to be maximised:

    accuracy   overall correctness on the held-out test set
    f1         harmonic mean of precision and recall for the AI class

Three are cost criteria, to be minimised:

    latency    median milliseconds to classify one piece of text
    size       bytes the trained model occupies on disk
    lime_time  seconds to produce one LIME explanation

Why cost criteria belong in the decision at all
-----------------------------------------------
Ranking models on accuracy variants alone would be a fake
multi-criteria problem. The test set is exactly balanced (300 human /
300 AI), so accuracy is *identically* the mean of sensitivity and
specificity, and F1 tracks the same quantities. Four such criteria are
one dimension wearing four hats, and TOPSIS over them would just
reproduce the accuracy ordering.

Latency, size and explanation time genuinely conflict with accuracy: the
transformer is expected to be the most accurate and the slowest, largest
and most expensive to explain, by orders of magnitude. That conflict is
what gives a multi-criteria method something to decide.

`lime_time` is the criterion that ties the two halves of this project
together: in a system whose purpose is explainable classification, the
cost of producing an explanation is a real property of the model, not an
implementation detail.

Why each measurement is taken the way it is
-------------------------------------------
Latency is measured one text at a time, not in batches. Batched
inference is much faster per item, but demo.py and explain.py classify
single inputs, so single-text latency is what a user actually
experiences.

Every timing is repeated and reported as (min, median, max) rather than
a single number, because timings vary between runs, and quality metrics
carry bootstrap confidence intervals for the same reason: a score from
600 test rows is an estimate, not an exact value.

Each measurement is also recorded as a linguistic category — the input
fuzzy TOPSIS operates on — by rating it against the threshold bands in
fuzzify.py.
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from fuzzify import classify, fuzzify
from metrics import bootstrap_ci, confusion, evaluate, format_table

from paths import (BASELINE_MODEL, DECISION_MATRIX, DISTILBERT_DIR, SVM_MODEL,
                   TEST_CSV, XGBOOST_MODEL, ensure_dirs)

# Criterion order is fixed here and relied on by select_model.py.
CRITERIA = ["accuracy", "f1", "latency_ms", "size_mb", "lime_seconds"]
DIRECTIONS = ["benefit", "benefit", "cost", "cost", "cost"]

LATENCY_TEXTS = 100      # texts per latency repeat
LATENCY_REPEATS = 5      # repeats, to get a (min, median, max) spread
LIME_TEXTS = 5           # explanations timed per model
LIME_SAMPLES = 1000      # must match explain.NUM_SAMPLES to be representative


def dir_size_bytes(path):
    """Total size on disk, recursing into a directory if needed."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def measure_latency(pipeline, texts):
    """
    Median milliseconds for a single-text prediction, repeated.

    Returns (min, median, max) across repeats. One warm-up pass is
    discarded first: the first call to a transformer pays for lazy
    kernel compilation and memory allocation on MPS, which would
    otherwise dominate the measurement.
    """
    pipeline.predict([texts[0]])  # warm-up, discarded

    per_repeat = []
    for _ in range(LATENCY_REPEATS):
        times = []
        for text in texts:
            start = time.perf_counter()
            pipeline.predict([text])
            times.append((time.perf_counter() - start) * 1000.0)
        per_repeat.append(float(np.median(times)))

    return (min(per_repeat), float(np.median(per_repeat)), max(per_repeat))


def measure_lime(pipeline, texts):
    """
    Seconds to produce one LIME explanation, as (min, median, max).

    Imported lazily so that a failure to build LIME explanations does not
    prevent the quality metrics from being collected.
    """
    from explain import explain_text

    times = []
    for text in texts:
        start = time.perf_counter()
        explain_text(text, pipeline=pipeline, num_samples=LIME_SAMPLES)
        times.append(time.perf_counter() - start)

    return (min(times), float(np.median(times)), max(times))


def load_models():
    """
    Every candidate, as (name, pipeline, path_on_disk).

    The transformer is imported inside the function so the classical
    models can still be benchmarked on a machine without torch
    installed.
    """
    from advanced_model import load_advanced_model

    return [
        ("LogisticRegression", joblib.load(BASELINE_MODEL), BASELINE_MODEL),
        ("LinearSVM", joblib.load(SVM_MODEL), SVM_MODEL),
        ("XGBoost", joblib.load(XGBOOST_MODEL), XGBOOST_MODEL),
        ("DistilBERT", load_advanced_model(), DISTILBERT_DIR),
    ]


def main():
    ensure_dirs()
    test_df = pd.read_csv(TEST_CSV)
    texts = test_df["text"].astype(str).tolist()
    y_true = test_df["label"].values

    # Fixed subsets so every model is timed on identical inputs.
    latency_texts = texts[:LATENCY_TEXTS]
    lime_texts = texts[:LIME_TEXTS]

    records = {}
    for name, pipeline, path in load_models():
        print(f"\n{'=' * 60}\nBenchmarking {name}\n{'=' * 60}")

        predictions = pipeline.predict(texts)
        scores = evaluate(y_true, predictions)
        print(f"  accuracy    {scores['accuracy']:.4f}")
        print(f"  f1          {scores['f1']:.4f}")
        print(f"  sensitivity {scores['sensitivity']:.4f}")
        print(f"  specificity {scores['specificity']:.4f}")

        acc_tfn = bootstrap_ci(y_true, predictions, "accuracy")
        f1_tfn = bootstrap_ci(y_true, predictions, "f1")
        print(f"  accuracy 95% CI  [{acc_tfn[0]:.4f}, {acc_tfn[2]:.4f}]")

        print("  timing latency...")
        latency_tfn = measure_latency(pipeline, latency_texts)
        print(f"  latency     {latency_tfn[1]:.2f} ms  (single text)")

        size_bytes = dir_size_bytes(path)
        size_mb = size_bytes / (1024 * 1024)
        print(f"  size        {size_mb:.2f} MB")

        print("  timing LIME explanations...")
        lime_tfn = measure_lime(pipeline, lime_texts)
        print(f"  lime        {lime_tfn[1]:.2f} s per explanation")

        crisp = {
            "accuracy": scores["accuracy"],
            "f1": scores["f1"],
            "latency_ms": latency_tfn[1],
            "size_mb": size_mb,
            "lime_seconds": lime_tfn[1],
        }
        categories = {c: classify(c, v) for c, v in crisp.items()}
        print("  rated       " + ", ".join(
            f"{c}={categories[c]}" for c in CRITERIA))

        records[name] = {
            "metrics": scores,
            "confusion": confusion(y_true, predictions).tolist(),
            "crisp": crisp,
            # Linguistic ratings and the TFN each one looks up. This is
            # what fuzzy TOPSIS consumes.
            "linguistic": {
                c: {"category": categories[c], "tfn": list(fuzzify(c, crisp[c]))}
                for c in CRITERIA
            },
            # Measurement spread, reported alongside the results rather
            # than fed into the ranking: bootstrap intervals for the
            # quality metrics, observed (min, median, max) for timings.
            "intervals": {
                "accuracy": list(acc_tfn),
                "f1": list(f1_tfn),
                "latency_ms": list(latency_tfn),
                "lime_seconds": list(lime_tfn),
            },
        }

    payload = {
        "criteria": CRITERIA,
        "directions": DIRECTIONS,
        "models": records,
    }
    with open(DECISION_MATRIX, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n\n{'=' * 60}\nDecision matrix\n{'=' * 60}")
    rows = [
        [name] + [f"{rec['crisp'][c]:.4f}" for c in CRITERIA]
        for name, rec in records.items()
    ]
    print(format_table(rows, ["model"] + CRITERIA))
    print(f"\nSaved to {DECISION_MATRIX}")


if __name__ == "__main__":
    main()
