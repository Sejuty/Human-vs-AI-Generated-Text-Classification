"""Runs every candidate model over the same example sentences and prints their
predictions and LIME explanations side by side."""

import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from explain import CLASS_NAMES, EXAMPLES, LOADERS, explain_text
from metrics import format_table

from paths import EXPLANATIONS_DIR, ensure_dirs

NUM_SAMPLES = 1000
TOP_WORDS = 4


def load_all():
    """Load every candidate, skipping any that has not been trained yet."""
    models = {}
    for key, loader in LOADERS.items():
        try:
            models[key] = loader()
        except (FileNotFoundError, OSError) as exc:
            print(f"[!] skipping {key}: {exc}")
    return models


def format_score(value):
    """Format a LIME weight without collapsing small ones to "+0.00"."""
    if value == 0:
        return "+0"
    if abs(value) < 0.005:
        return f"{value:+.1e}"
    return f"{value:+.2f}"


def top_words(scores, n=TOP_WORDS):
    """The n most influential words, strongest first, as a compact string."""
    ranked = sorted(scores, key=lambda ws: -abs(ws[1]))[:n]
    return ", ".join(f"{w}{format_score(s)}" for w, s in ranked)


def main():
    models = load_all()
    if not models:
        print("No trained models found. Run the training scripts first.")
        return

    ensure_dirs()
    disagreements = []

    for i, (label, text) in enumerate(EXAMPLES, start=1):
        print("=" * 78)
        print(f"Example {i} ({label})")
        print(f"  {text}")
        print("=" * 78)

        rows = []
        predictions = {}
        for key, pipeline in models.items():
            probabilities = pipeline.predict_proba([text])[0]
            predicted = int(pipeline.predict([text])[0])
            predictions[key] = predicted

            scores = explain_text(text, pipeline=pipeline, num_samples=NUM_SAMPLES)
            rows.append([
                key,
                CLASS_NAMES[predicted],
                f"{probabilities[predicted] * 100:.1f}%",
                top_words(scores),
            ])

        print(format_table(rows, ["model", "prediction", "confidence", "top LIME words"]))

        # Any split verdict is worth keeping: write each model's
        # highlighted explanation so the difference can be inspected.
        if len(set(predictions.values())) > 1:
            disagreements.append((i, label, text, dict(predictions)))
            print("\n  ** models disagree — writing highlighted explanations **")
            for key, pipeline in models.items():
                path = str(EXPLANATIONS_DIR / f"disagree_{i:02d}_{key}.html")
                explain_text(text, pipeline=pipeline,
                             num_samples=NUM_SAMPLES, save_html=path)
                print(f"     {path}")
        print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if disagreements:
        print(f"\n{len(disagreements)} of {len(EXAMPLES)} examples split the models:\n")
        for i, label, text, preds in disagreements:
            verdicts = ", ".join(f"{k}={CLASS_NAMES[v]}" for k, v in preds.items())
            print(f"  {i}. ({label}) {text[:56]}...")
            print(f"     {verdicts}")
        print(f"\nHighlighted explanations for these are in {EXPLANATIONS_DIR}")
    else:
        print("\nAll models agreed on every example.")


if __name__ == "__main__":
    main()
