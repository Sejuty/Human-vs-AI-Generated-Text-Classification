"""Measures every criterion for every candidate model and writes the decision
matrix that select_model.py ranks."""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from fuzzify import classify, fuzzify
from metrics import bootstrap_ci, confusion, evaluate, format_table

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from paths import (BASELINE_MODEL, CONTENT_MODEL, DECISION_MATRIX,
                   DISTILBERT_DIR, SVM_MODEL, TEST_CSV, XGBOOST_MODEL,
                   ensure_dirs)

# Criterion order is fixed here and relied on by select_model.py.
CRITERIA = ["accuracy", "f1", "latency_ms", "size_mb", "lime_seconds",
            "content_share"]
DIRECTIONS = ["benefit", "benefit", "cost", "cost", "cost", "benefit"]

# Measured and reported, but excluded from the ranking: this model scores
# ~100% on content_share by construction, having no stop-word features.
ABLATIONS = {"LogisticRegression-Content"}

LATENCY_TEXTS = 100      # texts per latency repeat
LATENCY_REPEATS = 5      # repeats, to get a (min, median, max) spread
LIME_TEXTS = 10          # explanations per model, for timing and content share
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
    """Median milliseconds for a single-text prediction, repeated."""
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
    """Returns ((min, median, max) seconds per explanation, mean content
    share)."""
    from explain import explain_text

    times, shares = [], []
    for text in texts:
        start = time.perf_counter()
        weights = explain_text(text, pipeline=pipeline, num_samples=LIME_SAMPLES)
        times.append(time.perf_counter() - start)

        total = sum(abs(w) for _, w in weights)
        content = sum(abs(w) for word, w in weights
                      if word.lower() not in ENGLISH_STOP_WORDS)
        shares.append(100.0 * content / total if total else 0.0)

    return (min(times), float(np.median(times)), max(times)), float(np.mean(shares))


def load_models():
    """Every candidate, as (name, pipeline, path_on_disk)."""
    from advanced_model import load_advanced_model

    return [
        ("LogisticRegression", joblib.load(BASELINE_MODEL), BASELINE_MODEL),
        ("LinearSVM", joblib.load(SVM_MODEL), SVM_MODEL),
        ("XGBoost", joblib.load(XGBOOST_MODEL), XGBOOST_MODEL),
        ("LogisticRegression-Content", joblib.load(CONTENT_MODEL), CONTENT_MODEL),
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
        lime_tfn, content_share = measure_lime(pipeline, lime_texts)
        print(f"  lime        {lime_tfn[1]:.2f} s per explanation")
        print(f"  content     {content_share:.1f}% of explanation weight")

        crisp = {
            "accuracy": scores["accuracy"],
            "f1": scores["f1"],
            "latency_ms": latency_tfn[1],
            "size_mb": size_mb,
            "lime_seconds": lime_tfn[1],
            "content_share": content_share,
        }
        categories = {c: classify(c, v) for c, v in crisp.items()}
        print("  rated       " + ", ".join(
            f"{c}={categories[c]}" for c in CRITERIA))

        records[name] = {
            "ablation": name in ABLATIONS,
            "metrics": scores,
            "confusion": confusion(y_true, predictions).tolist(),
            "crisp": crisp,
            # Linguistic ratings and the TFN each one looks up. This is
            # what fuzzy TOPSIS consumes.
            "linguistic": {
                c: {"category": categories[c], "tfn": list(fuzzify(c, crisp[c]))}
                for c in CRITERIA
            },
            # Measurement spread.
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
