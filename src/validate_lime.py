"""Checks that the LIME explanations this project relies on are actually
describing the models, rather than merely looking plausible."""

import json
import os
import re
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

from explain import NUM_SAMPLES, explain_text
from metrics import format_table

from paths import (BASELINE_MODEL, CONTENT_MODEL, LIME_VALIDATION, SVM_MODEL,
                   TEST_CSV, XGBOOST_MODEL, ensure_dirs)

N_FAITHFULNESS = 8   # texts used for the deletion test
N_STABILITY = 4      # texts re-explained under several seeds
SEEDS = (0, 1, 2)    # seeds compared for stability
TOP_K = 5            # words deleted in the faithfulness test

# Minimum mean probability drop for the deletion test to count as passing.
FAITHFULNESS_FLOOR = 0.05

# Phrasing characteristic of assistant writing, for the register check.
AI_REGISTER = ["it is important", "as an ai", "in conclusion", "keep in mind"]


def delete_words(text, words):
    """Remove whole-word occurrences, leaving the rest of the text intact."""
    out = text
    for word in words:
        out = re.sub(rf"\b{re.escape(word)}\b", " ", out)
    return re.sub(r"\s+", " ", out).strip()


def supporting_words(weights, predicted, k=TOP_K):
    """The k words most supporting the predicted class."""
    signed = [(w, s if predicted == 1 else -s) for w, s in weights]
    return [w for w, s in sorted(signed, key=lambda ws: -ws[1])[:k] if s > 0]


def faithfulness(pipeline, texts, num_samples=NUM_SAMPLES):
    """Mean drop in predicted-class probability after deleting top words."""
    drops, sharpness, confidences = [], [], []
    for text in texts:
        weights = explain_text(text, pipeline=pipeline, num_features=8,
                               num_samples=num_samples)
        predicted = int(pipeline.predict([text])[0])
        before = float(pipeline.predict_proba([text])[0][predicted])

        removed = supporting_words(weights, predicted)
        if not removed:
            continue
        after = float(pipeline.predict_proba([delete_words(text, removed)])[0][predicted])

        drops.append(before - after)
        sharpness.append(max(abs(s) for _, s in weights))
        confidences.append(before)

    return {
        "mean_drop": float(np.mean(drops)),
        "min_drop": float(np.min(drops)),
        "n_texts": len(drops),
        "median_sharpness": float(np.median(sharpness)),
        "mean_confidence": float(np.mean(confidences)),
        # Negative correlation is the expected sign: the more confident
        # the model, the flatter the local explanation.
        "confidence_sharpness_r": (
            float(np.corrcoef(confidences, sharpness)[0, 1])
            if len(set(np.round(confidences, 6))) > 1 else float("nan")
        ),
    }


def stability(pipeline, texts, seeds=SEEDS, num_samples=NUM_SAMPLES):
    """Mean pairwise correlation of word weights across perturbation seeds."""
    correlations, overlaps = [], []
    for text in texts:
        runs = [dict(explain_text(text, pipeline=pipeline, num_features=10,
                                  num_samples=num_samples, seed=s))
                for s in seeds]
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                shared = set(runs[i]) & set(runs[j])
                overlaps.append(len(shared) / max(len(runs[i]), len(runs[j])))
                if len(shared) > 2:
                    a = [runs[i][w] for w in shared]
                    b = [runs[j][w] for w in shared]
                    if np.std(a) > 0 and np.std(b) > 0:
                        correlations.append(np.corrcoef(a, b)[0, 1])

    return {
        "mean_weight_correlation": float(np.mean(correlations)) if correlations else float("nan"),
        "min_weight_correlation": float(np.min(correlations)) if correlations else float("nan"),
        "mean_word_overlap": float(np.mean(overlaps)),
        "n_seeds": len(seeds),
        "n_texts": len(texts),
    }


def register_check(pipeline, texts, num_samples=NUM_SAMPLES):
    """Do AI-register phrases carry AI-ward weight where they appear?"""
    hits, total = 0, 0
    for text in texts:
        if int(pipeline.predict([text])[0]) != 1:
            continue
        lowered = text.lower()
        present = [p for p in AI_REGISTER if p in lowered]
        if not present:
            continue
        weights = dict(explain_text(text, pipeline=pipeline, num_features=12,
                                    num_samples=num_samples))
        for phrase in present:
            for token in phrase.split():
                if token in weights:
                    total += 1
                    hits += weights[token] > 0
    return {"positive": hits, "checked": total,
            "rate": (hits / total) if total else float("nan")}


def load_models():
    from advanced_model import load_advanced_model
    from explain import load_model
    return [
        ("LogisticRegression", lambda: load_model(BASELINE_MODEL)),
        ("LinearSVM", lambda: load_model(SVM_MODEL)),
        ("XGBoost", lambda: load_model(XGBOOST_MODEL)),
        ("LogisticRegression-Content", lambda: load_model(CONTENT_MODEL)),
        ("DistilBERT", load_advanced_model),
    ]


def main():
    ensure_dirs()
    test_df = pd.read_csv(TEST_CSV)
    texts = test_df["text"].astype(str).tolist()

    # Fixed slices so every model is measured on identical inputs.
    faith_texts = texts[:N_FAITHFULNESS]
    stab_texts = texts[:N_STABILITY]
    reg_texts = texts[:20]

    results = {}
    for name, loader in load_models():
        print(f"\n{'=' * 66}\n{name}\n{'=' * 66}")
        pipeline = loader()

        print("  deletion test...")
        f = faithfulness(pipeline, faith_texts)
        print(f"    mean probability drop   {f['mean_drop']:+.4f}  "
              f"(min {f['min_drop']:+.4f} over {f['n_texts']} texts)")
        print(f"    median peak |weight|    {f['median_sharpness']:.4f}")
        print(f"    mean confidence         {f['mean_confidence']:.4f}")

        print("  stability across seeds...")
        s = stability(pipeline, stab_texts)
        print(f"    weight correlation      {s['mean_weight_correlation']:.3f}  "
              f"(min {s['min_weight_correlation']:.3f})")
        print(f"    word overlap            {s['mean_word_overlap']:.3f}")

        print("  register check...")
        r = register_check(pipeline, reg_texts)
        rate = "n/a" if np.isnan(r["rate"]) else f"{r['rate']:.2f}"
        print(f"    AI-ward rate            {rate}  ({r['positive']}/{r['checked']})")

        results[name] = {"faithfulness": f, "stability": s, "register": r}

    # ---- Summary -------------------------------------------------------
    print(f"\n\n{'=' * 66}\nSUMMARY\n{'=' * 66}\n")
    print(format_table(
        [[n,
          f"{v['faithfulness']['mean_drop']:+.4f}",
          f"{v['faithfulness']['median_sharpness']:.4f}",
          f"{v['stability']['mean_weight_correlation']:.3f}",
          f"{v['faithfulness']['mean_confidence']:.3f}"]
         for n, v in results.items()],
        ["model", "faithfulness", "sharpness", "stability", "confidence"]))

    with open(LIME_VALIDATION, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWritten to {LIME_VALIDATION}")

    # ---- Assertions ----------------------------------------------------
    print(f"\n{'=' * 66}\nCHECKS\n{'=' * 66}")
    failures = []
    for name, v in results.items():
        drop = v["faithfulness"]["mean_drop"]
        if drop < FAITHFULNESS_FLOOR:
            failures.append(
                f"{name}: mean probability drop {drop:+.4f} is below "
                f"{FAITHFULNESS_FLOOR} — the highlighted words do not appear "
                f"to be what the model is using")
        else:
            print(f"  {name:20} deletion test passed  ({drop:+.4f})")

        corr = v["stability"]["mean_weight_correlation"]
        if not np.isnan(corr) and corr < 0.5:
            failures.append(
                f"{name}: cross-seed weight correlation {corr:.3f} is low — "
                f"explanations are dominated by sampling noise")

    if failures:
        print()
        for f in failures:
            print(f"  FAIL  {f}")
        raise SystemExit(1)

    print("\nLIME is behaving as required on every model.")


if __name__ == "__main__":
    main()
