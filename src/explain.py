"""Loads the trained Human-vs-AI classifier and explains predictions using LIME
(Local Interpretable Model-agnostic Explanations).

Explains one passage by default, or the built-in demonstration set with --demo."""

import html
import json
import os
import re
import warnings

import joblib
from lime.lime_text import LimeTextExplainer

from paths import (BASELINE_MODEL, CONTENT_MODEL, EXPLANATIONS_DIR,
                   SELECTED_JSON, SVM_MODEL, XGBOOST_MODEL, DATA_DIR,
                   ensure_dirs)

# Silence harmless numeric RuntimeWarnings sklearn prints while LIME
# scores its many tiny perturbed samples, so live output stays clean.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Perturbed samples per explanation. LIME defaults to 5000; each one is a
# forward pass, so that default costs ~30s against DistilBERT.
NUM_SAMPLES = 1000

# Fixes LIME's perturbation sampling so a given text always yields the same
# explanation.
SEED = 42

# Matches label encoding used in data_prep.py/train.py: 0 = human, 1 = AI.
CLASS_NAMES = ["Human", "AI-generated"]

# Short hand-written sentences: clearly human.
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

# Three excerpts drawn from the held-out HC3 test set.
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


def load_model(path):
    return joblib.load(path)


# Every loader returns something with .predict / .predict_proba over raw
# strings.
LOADERS = {
    "baseline": lambda: load_model(BASELINE_MODEL),
    "svm": lambda: load_model(SVM_MODEL),
    "xgboost": lambda: load_model(XGBOOST_MODEL),
    "content": lambda: load_model(CONTENT_MODEL),
    "advanced": lambda: __import__("advanced_model").load_advanced_model(),
}

# Maps the model names recorded by select_model.py onto the CLI keys.
SELECTED_TO_KEY = {
    "LogisticRegression": "baseline",
    "LinearSVM": "svm",
    "XGBoost": "xgboost",
    "LogisticRegression-Content": "content",
    "DistilBERT": "advanced",
}


def resolve_selected():
    """Which model did the multi-criteria ranking pick?"""
    if not os.path.exists(SELECTED_JSON):
        print(f"[!] {SELECTED_JSON} not found — run select_model.py to choose a "
              f"model by ranking. Falling back to the baseline.\n")
        return "baseline"

    with open(SELECTED_JSON) as fh:
        selection = json.load(fh)

    name = selection["selected"]
    key = SELECTED_TO_KEY.get(name, "baseline")
    print(f"Model: {name} ({_ranking_note(selection, name)})\n")
    return key


def _ranking_note(selection, name):
    """One-line provenance: where `name` placed under each ranking."""
    ties = [group for group in selection.get("fuzzy_ties", []) if name in group]

    parts = []
    for label, order in (("TOPSIS", selection.get("topsis_ranking")),
                         ("fuzzy TOPSIS", selection.get("fuzzy_ranking"))):
        if not order or name not in order:
            continue
        group = ties[0] if (label == "fuzzy TOPSIS" and ties) else [name]
        rank = min(order.index(n) for n in group if n in order) + 1
        tied = "tied " if len(group) > 1 else ""
        parts.append(f"{label} {tied}rank {rank}")

    return "; ".join(parts) if parts else "selected by multi-criteria ranking"


def load_selected_model():
    """Load whichever model the multi-criteria ranking selected."""
    return LOADERS[resolve_selected()]()


# LIME reports word weights *relative to a chosen class*.
AI_LABEL = 1


def explain_text(text, num_features=8, pipeline=None,
                 num_samples=NUM_SAMPLES, save_html=None, seed=SEED):
    """Run LIME on `text` and return the top word-level contributions as a list
    of (word, importance_score) tuples."""
    if pipeline is None:
        pipeline = load_selected_model()

    explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=seed)
    explanation = explainer.explain_instance(
        text,
        pipeline.predict_proba,
        num_features=num_features,
        labels=[AI_LABEL],
        num_samples=num_samples,
    )

    if save_html:
        os.makedirs(os.path.dirname(save_html) or ".", exist_ok=True)
        explanation.save_to_file(save_html)

    return explanation.as_list(label=AI_LABEL)


def analyze(text, num_features=8, pipeline=None, num_samples=NUM_SAMPLES,
            save_html=None, label=None):
    """Prediction, confidence and LIME word weights for one text, in the shape
    both the terminal output and the HTML report want."""
    if pipeline is None:
        pipeline = load_selected_model()

    probabilities = pipeline.predict_proba([text])[0]
    predicted = int(pipeline.predict([text])[0])

    return {
        "label": label,
        "text": text,
        "predicted": predicted,
        "confidence": float(probabilities[predicted]),
        "weights": explain_text(text, num_features=num_features,
                                pipeline=pipeline, num_samples=num_samples,
                                save_html=save_html),
    }


def print_explanation(text, num_features=8, pipeline=None,
                      num_samples=NUM_SAMPLES, save_html=None):
    """Print one prediction and the words driving it. Returns the analysis so
    callers building a report do not have to explain the text twice."""
    result = analyze(text, num_features=num_features, pipeline=pipeline,
                     num_samples=num_samples, save_html=save_html)

    print(f"Text: {text}")
    print(f"Prediction: {CLASS_NAMES[result['predicted']]} "
          f"({result['confidence'] * 100:.1f}% confidence)")

    word_scores = result["weights"]
    ai_words = [(w, s) for w, s in word_scores if s > 0]
    human_words = [(w, s) for w, s in word_scores if s < 0]

    print("\nWords pushing toward AI-generated:")
    if ai_words:
        for word, score in sorted(ai_words, key=lambda x: -x[1]):
            print(f"  {word:15s} {score:+.4f}")
    else:
        print("  (none)")

    print("\nWords pushing toward Human:")
    if human_words:
        for word, score in sorted(human_words, key=lambda x: x[1]):
            print(f"  {word:15s} {score:+.4f}")
    else:
        print("  (none)")

    return result


# ---- HTML report --------------------------------------------------------
#
# LIME's own save_to_file() writes a 1.2MB d3 page per text, with no record of
# what was classified. This renders every example onto one self-contained page
# instead: no JavaScript, no CDN, nothing to install.

REPORT_CSS = """
:root {
  --bg: #fbfbfa; --card: #fff; --ink: #1c1c1a; --muted: #6b6b66;
  --line: #e4e4e0; --ai: #b4441f; --human: #1f6f5c;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16181c; --card: #1e2126; --ink: #e8e8e4; --muted: #9a9a94;
    --line: #2f333a; --ai: #ff9470; --human: #5fd3b2;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 40px 20px; background: var(--bg); color: var(--ink);
  font: 15px/1.6 ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif;
}
main { max-width: 780px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
.sub { color: var(--muted); font-size: 13px; margin: 0 0 32px; }
.card {
  background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 20px 22px; margin-bottom: 18px;
}
.chip {
  display: inline-block; font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.07em; color: var(--muted); border: 1px solid var(--line);
  border-radius: 99px; padding: 2px 10px; margin-bottom: 14px;
}
.passage { margin: 0 0 18px; }
mark {
  background: none; color: inherit; border-radius: 3px; padding: 0 1px;
  box-shadow: inset 0 -0.55em 0 rgba(var(--tint), var(--a));
}
mark.ai { --tint: 220, 90, 40; }
mark.human { --tint: 40, 160, 130; }
@media (prefers-color-scheme: dark) {
  mark.ai { --tint: 255, 140, 100; }
  mark.human { --tint: 95, 211, 178; }
}
.verdict { display: flex; align-items: baseline; gap: 10px; font-size: 14px; }
.verdict b { font-weight: 600; }
.verdict.ai b { color: var(--ai); }
.verdict.human b { color: var(--human); }
.verdict .pct { color: var(--muted); font-variant-numeric: tabular-nums; }
.meter {
  height: 4px; border-radius: 2px; background: var(--line);
  margin: 8px 0 20px; overflow: hidden;
}
.meter i { display: block; height: 100%; }
.meter i.ai { background: var(--ai); }
.meter i.human { background: var(--human); }
table.weights { width: 100%; border-collapse: collapse; font-size: 13px; }
table.weights td { padding: 3px 0; vertical-align: middle; }
td.word {
  width: 26%; text-align: right; padding-right: 12px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
td.track { position: relative; }
td.track div { position: absolute; top: 50%; height: 9px; margin-top: -4.5px; }
td.track div.ai { left: 50%; background: var(--ai); border-radius: 0 2px 2px 0; }
td.track div.human { right: 50%; background: var(--human); border-radius: 2px 0 0 2px; }
td.track::before {
  content: ""; position: absolute; left: 50%; top: 0; bottom: 0;
  border-left: 1px solid var(--line);
}
td.score {
  width: 66px; text-align: right; color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.legend { color: var(--muted); font-size: 12px; margin-top: 26px; }
.legend b.ai { color: var(--ai); }
.legend b.human { color: var(--human); }
"""


def _highlight(text, weights):
    """The passage with LIME's words tinted — AI-ward one colour, human-ward the
    other, opacity scaled by weight relative to the strongest word here."""
    lookup = {word.lower(): score for word, score in weights}
    peak = max((abs(s) for _, s in weights), default=0.0)
    if not peak:
        return html.escape(text)

    out = []
    # Split on non-word runs but keep them, so the passage reads unchanged.
    for token in re.split(r"(\W+)", text):
        score = lookup.get(token.lower())
        if score is None or not token:
            out.append(html.escape(token))
            continue
        side = "ai" if score > 0 else "human"
        # Floor the opacity so the weakest words are still visibly marked.
        alpha = 0.18 + 0.62 * (abs(score) / peak)
        out.append(f'<mark class="{side}" style="--a:{alpha:.2f}" '
                   f'title="{score:+.4f}">{html.escape(token)}</mark>')
    return "".join(out)


def _weight_table(weights):
    """Word weights as a bar chart running out from a centre line."""
    peak = max((abs(s) for _, s in weights), default=0.0)
    rows = []
    for word, score in sorted(weights, key=lambda ws: -abs(ws[1])):
        side = "ai" if score > 0 else "human"
        width = (abs(score) / peak * 50) if peak else 0
        rows.append(
            f'<tr><td class="word">{html.escape(word)}</td>'
            f'<td class="track"><div class="{side}" style="width:{width:.1f}%"></div></td>'
            f'<td class="score">{score:+.4f}</td></tr>'
        )
    return f'<table class="weights">{"".join(rows)}</table>'


def _card(result):
    side = "ai" if result["predicted"] == AI_LABEL else "human"
    chip = (f'<div class="chip">{html.escape(result["label"])}</div>'
            if result["label"] else "")
    return (
        f'<section class="card">{chip}'
        f'<p class="passage">{_highlight(result["text"], result["weights"])}</p>'
        f'<div class="verdict {side}"><span>Prediction:</span>'
        f'<b>{CLASS_NAMES[result["predicted"]]}</b>'
        f'<span class="pct">{result["confidence"] * 100:.1f}% confidence</span></div>'
        f'<div class="meter"><i class="{side}" '
        f'style="width:{result["confidence"] * 100:.1f}%"></i></div>'
        f'{_weight_table(result["weights"])}</section>'
    )


def render_report(results, path, model_name=""):
    """Write every analysis onto one self-contained page."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    subtitle = f"{model_name} · " if model_name else ""
    plural = "s" if len(results) != 1 else ""
    body = "".join(_card(r) for r in results)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>LIME explanations</title><style>{REPORT_CSS}</style></head>'
            f'<body><main><h1>Human vs AI — LIME explanations</h1>'
            f'<p class="sub">{html.escape(subtitle)}{len(results)} passage{plural}</p>'
            f'{body}'
            f'<p class="legend">Shading marks the words LIME found most '
            f'influential: <b class="ai">toward AI-generated</b>, '
            f'<b class="human">toward human</b>. Hover a word for its weight.</p>'
            f'</main></body></html>'
        )
    return path


# Stands in for --html given without a path.
AUTO_HTML = object()


def _report_path(requested, key, stem, default=False):
    """Where the report goes: the path asked for, or one named for the model.
    `default` writes a report even when --html was not passed at all."""
    if requested is AUTO_HTML or (requested is None and default):
        return str(EXPLANATIONS_DIR / f"{key}_{stem}.html")
    return requested


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?",
                        help="text to explain, or a path to a file containing it "
                             "(default: data/sample_text.txt)")
    parser.add_argument("--demo", action="store_true",
                        help="explain the built-in example set and write an "
                             "HTML report to explanations/")
    parser.add_argument("--model", choices=["selected"] + list(LOADERS),
                        default="selected",
                        help="which model to explain (default: the selected one)")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                        help="LIME perturbation samples")
    parser.add_argument("--html", nargs="?", const=AUTO_HTML, metavar="PATH",
                        help="also write the HTML report; bare, it is named for "
                             "the model and dropped in explanations/")
    args = parser.parse_args()

    if args.demo and args.text:
        parser.error("--demo runs the built-in examples; do not also pass a text")

    ensure_dirs()
    key = resolve_selected() if args.model == "selected" else args.model
    pipeline = LOADERS[key]()

    if args.demo:
        results = []
        for label, text in EXAMPLES:
            print("=" * 60)
            print(f"Example ({label})")
            print("-" * 60)
            result = print_explanation(text, pipeline=pipeline,
                                       num_samples=args.num_samples)
            result["label"] = label
            results.append(result)
            print()
        path = _report_path(args.html, key, "demo", default=True)
    else:
        source = args.text or str(DATA_DIR / "sample_text.txt")
        if os.path.isfile(source):
            with open(source, encoding="utf-8") as fh:
                text = fh.read().strip()
        else:
            text = source
        results = [print_explanation(text, pipeline=pipeline,
                                     num_samples=args.num_samples)]
        path = _report_path(args.html, key, "custom")

    if path:
        render_report(results, path, model_name=key)
        print(f"\nHighlighted explanations written to {path}")


if __name__ == "__main__":
    main()
