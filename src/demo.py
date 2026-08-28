"""
demo.py
-------
CLI demo script for the live presentation.

Usage:
    python demo.py                            # all examples, selected model
    python demo.py "some sentence to check"   # one custom sentence
    python demo.py --model baseline           # pick a specific model
    python demo.py --html                     # also write LIME HTML files

By default this loads whichever model the multi-criteria ranking in
select_model.py chose, recorded in models/selected.json. Pass --model to
override that and run any single candidate, which is what makes a
side-by-side comparison possible during the demo.

Every model here — the TF-IDF baseline, the SVM, XGBoost, and the
fine-tuned transformer — exposes the same predict/predict_proba
interface, so nothing below this line changes depending on which one is
loaded. That is the whole reason the LIME layer works unmodified across
all four.
"""

import argparse
import json
import os
import re
import sys
import warnings

# Silence harmless numeric RuntimeWarnings sklearn prints while LIME
# scores its many tiny perturbed samples, so live output stays clean.
warnings.filterwarnings("ignore", category=RuntimeWarning)

from explain import NUM_SAMPLES, load_model, print_explanation

from paths import (BASELINE_MODEL, EXPLANATIONS_DIR, SELECTED_JSON, SVM_MODEL,
                   XGBOOST_MODEL, ensure_dirs)

# Short hand-written sentences: clearly human, clearly AI-style, and a
# couple of ambiguous ones, so the live demo has a reliable script to
# fall back on if nothing is typed. The human/AI
# ones were checked against the trained model beforehand so the live
# demo doesn't hinge on a borderline prediction; the ambiguous ones
# are left as-is since their point is to show the model being unsure.
SHORT_EXAMPLES = [
    ("human", "lol idk man i just think people overthink this stuff way too much tbh"),
    ("human", "My grandma used to make this soup every winter and honestly nothing tastes the same anymore."),
    ("human", "ngl i skipped half the lecture but the exam wasnt that bad lol"),
    ("ai", "In conclusion, it is important to consider multiple perspectives when analyzing this complex issue."),
    ("ai", "It is important to note that individual results may vary depending on a variety of factors."),
    ("ai", "It is important to remember that this is a general overview and may not apply to every situation."),
    ("ambiguous", "The weather today is quite nice, perfect for a walk in the park."),
    ("ambiguous", "I appreciate your question and would be happy to help clarify this for you."),
]

# Three excerpts drawn from the held-out HC3 test set. HC3 answers run to
# roughly 865 characters on average, so these are in-distribution in a way
# the short sentences above are not — and the difference shows in the
# explanations. On a long, familiar passage LIME has enough signal to
# separate words clearly; on a one-line sentence a confident model returns
# almost flat weights, because removing any single word barely moves a
# probability already pinned near 1.
#
# Running both groups in one demo makes that contrast visible rather than
# leaving it as a claim in the report.
HC3_EXAMPLES = [
    ("hc3 human",
     "The problem is not in holding the World Cup , but in how it was "
     "done . If the government made the mobility improvements they "
     "promise ( better airports , more subways , better highways , etc "
     "... ) , and did n't wasted so much money creating stadiums on the "
     "middle of no where , had much less cities holding games , keep the "
     "promise of not using public money to build stadiums , maybe the "
     "World Cup would be a good thing and the economy in general would "
     "end on the black . However , they promise too much , delivered the "
     "bare minimum , wasted money and time , used public money to do "
     "almost everything , it was not worth to have the World Cup there ."),
    ("hc3 ai",
     "Equity refers to the ownership interest in a company. It "
     "represents the residual value that would be left over for "
     "shareholders if all of a company's debts were paid off. In other "
     "words, equity is the value of the company that is owned by the "
     "shareholders. This can be represented in the form of stock or "
     "shares in the company. Equity can also refer to the difference "
     "between the value of an asset, such as a home, and any debts or "
     "liabilities associated with that asset. For example, if you own a "
     "home worth $500,000 and you have a mortgage balance of $400,000, "
     "your equity in the home would be $100,000."),
    # True label is AI. The bag-of-words models and the transformer split
    # on this one, so it is worth having in the live set.
    ("hc3 contested",
     "Archie Manning is the father of three children: Cooper Manning, "
     "Peyton Manning, and Eli Manning. Cooper Manning is a former "
     "professional football player who was forced to retire due to a "
     "spinal condition. Peyton Manning is a former professional football "
     "player who is widely considered one of the greatest quarterbacks "
     "of all time. He won two Super Bowls, one with the Indianapolis "
     "Colts and one with the Denver Broncos. Eli Manning is also a "
     "former professional football player who played for the New York "
     "Giants. He is a two-time Super Bowl champion, having won Super "
     "Bowls XLII and XLVI with the Giants."),
]

EXAMPLES = SHORT_EXAMPLES + HC3_EXAMPLES


# Every loader returns something with .predict / .predict_proba over raw
# strings. joblib models are loaded eagerly; the transformer is imported
# lazily so the classical models still run without torch installed.
LOADERS = {
    "baseline": lambda: load_model(BASELINE_MODEL),
    "svm": lambda: load_model(SVM_MODEL),
    "xgboost": lambda: load_model(XGBOOST_MODEL),
    "advanced": lambda: __import__("advanced_model").load_advanced_model(),
}

# Maps the model names recorded by select_model.py onto the CLI keys.
SELECTED_TO_KEY = {
    "LogisticRegression": "baseline",
    "LinearSVM": "svm",
    "XGBoost": "xgboost",
    "DistilBERT": "advanced",
}


def resolve_selected():
    """
    Which model did the TOPSIS ranking pick? Falls back to the baseline
    if select_model.py has not been run yet, so the demo never hard-fails
    on a missing file.
    """
    if not os.path.exists(SELECTED_JSON):
        print(f"[!] {SELECTED_JSON} not found — run select_model.py to choose a "
              f"model by ranking. Falling back to the baseline.\n")
        return "baseline"
    with open(SELECTED_JSON) as fh:
        name = json.load(fh)["selected"]
    key = SELECTED_TO_KEY.get(name, "baseline")
    print(f"Using the model selected by multi-criteria ranking: {name}\n")
    return key


def run_one(pipeline, text, label=None, num_samples=NUM_SAMPLES, html_path=None):
    header = f"Example ({label})" if label else "Custom input"
    print("=" * 60)
    print(header)
    print("-" * 60)
    print_explanation(text, pipeline=pipeline, num_samples=num_samples,
                      save_html=html_path)
    if html_path:
        print(f"\nHighlighted explanation written to {html_path}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="a sentence to classify and explain")
    parser.add_argument("--model", choices=["selected"] + list(LOADERS),
                        default="selected",
                        help="which model to run (default: the one TOPSIS selected)")
    parser.add_argument("--html", action="store_true",
                        help="also write LIME highlighted-text HTML to explanations/")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                        help="LIME perturbation samples per explanation")
    args = parser.parse_args()

    ensure_dirs()
    key = resolve_selected() if args.model == "selected" else args.model
    pipeline = LOADERS[key]()

    def html_path(stem):
        if not args.html:
            return None
        # Labels contain spaces ("hc3 ai"); keep filenames shell-safe.
        slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
        return str(EXPLANATIONS_DIR / f"{key}_{slug}.html")

    if args.text:
        run_one(pipeline, args.text, num_samples=args.num_samples,
                html_path=html_path("custom"))
    else:
        for i, (label, text) in enumerate(EXAMPLES, start=1):
            run_one(pipeline, text, label=label, num_samples=args.num_samples,
                    html_path=html_path(f"{i:02d}_{label}"))


if __name__ == "__main__":
    main()
