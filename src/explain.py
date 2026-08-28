"""
explain.py
----------
Loads the trained Human-vs-AI classifier and explains a single
prediction using LIME (Local Interpretable Model-agnostic
Explanations).

How LIME works, briefly:
LIME treats the model as a black box. For one input text, it:
  1. Generates many perturbed versions of that text (by randomly
     removing words).
  2. Runs the real model on all those perturbed versions to see how
     the predicted probability changes.
  3. Fits a simple, interpretable linear model on those perturbations
     to approximate how the black-box model behaves *locally*, around
     this one piece of text.
  4. The weights of that local linear model become the word-level
     "importance scores". This module always reports them relative to
     the AI class, so a positive score means "pushes toward
     AI-generated" and a negative one means "pushes toward Human", no
     matter which class was actually predicted.

Because it's a *local* approximation, LIME explains one prediction at
a time — it isn't a global summary of the whole model.

Two knobs matter once this runs against a transformer rather than the
logistic-regression baseline:

  * `num_samples` controls how many perturbed variants LIME scores.
    LIME's own default is 5000, and every one of those is a full forward
    pass through the model. That is cheap for TF-IDF and slow enough to
    stall a live demo for DistilBERT, so the default here is 1000 —
    still enough for stable word rankings.
  * `save_html` writes LIME's native highlighted-text view, where the
    original passage is rendered with each word shaded by how much it
    pushed the prediction. That reads far better on a projector than a
    column of numbers.

Anything exposing sklearn's `predict` / `predict_proba` over raw strings
can be passed as `pipeline` — the TF-IDF baseline, the SVM, XGBoost, or
the TransformerPipeline wrapper in advanced_model.py.
"""

import os

import joblib
from lime.lime_text import LimeTextExplainer

from paths import BASELINE_MODEL

MODEL_PATH = BASELINE_MODEL

# Perturbed samples per explanation. LIME defaults to 5000; each one is
# a forward pass, so that default costs ~30s against DistilBERT. 1000
# keeps the top words stable while staying fast enough to run live.
NUM_SAMPLES = 1000

# Fixes LIME's perturbation sampling so a given text always yields the
# same explanation. Reproducibility, not robustness — how far an
# explanation moves when this changes is measured by validate_lime.py.
SEED = 42

# Matches label encoding used in data_prep.py/train.py: 0 = human, 1 = AI.
CLASS_NAMES = ["Human", "AI-generated"]


def load_model(path=MODEL_PATH):
    return joblib.load(path)


# LIME reports word weights *relative to a chosen class*. For binary
# classification the two are exact mirrors: a word worth +0.09 toward
# "AI" is worth -0.09 toward "Human". Everything here is therefore
# expressed relative to AI_LABEL, so the sign always means the same
# thing — positive pushes toward AI-generated, negative toward Human —
# regardless of which class the model actually predicted.
#
# Asking LIME for the *predicted* class instead would flip the sign
# convention every time the model happened to predict Human, and the
# callers below label the signs unconditionally.
AI_LABEL = 1


def explain_text(text, num_features=8, pipeline=None,
                 num_samples=NUM_SAMPLES, save_html=None, seed=SEED):
    """
    Run LIME on `text` and return the top word-level contributions as a
    list of (word, importance_score) tuples.

    Positive scores push toward "AI-generated", negative scores push
    toward "Human" — always, whichever class was predicted. See the note
    on AI_LABEL above for why that is not automatic.

    `num_samples` is how many perturbed variants LIME scores; lower is
    faster but noisier. `save_html`, if given a path, also writes LIME's
    highlighted-text rendering of this explanation there. `seed` fixes
    the perturbation sampling; varying it is how validate_lime.py
    measures how much an explanation moves between runs.
    """
    if pipeline is None:
        pipeline = load_model()

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


def print_explanation(text, num_features=8, pipeline=None,
                      num_samples=NUM_SAMPLES, save_html=None):
    """
    Print, in order:
      - the input text
      - the model's prediction (Human or AI-generated) with confidence %
      - top words pushing toward AI-generated (positive scores)
      - top words pushing toward Human (negative scores)
    """
    if pipeline is None:
        pipeline = load_model()

    probabilities = pipeline.predict_proba([text])[0]
    predicted_label = int(pipeline.predict([text])[0])
    confidence = probabilities[predicted_label] * 100

    print(f"Text: {text}")
    print(f"Prediction: {CLASS_NAMES[predicted_label]} ({confidence:.1f}% confidence)")

    word_scores = explain_text(
        text, num_features=num_features, pipeline=pipeline,
        num_samples=num_samples, save_html=save_html,
    )
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


if __name__ == "__main__":
    import sys

    text = sys.argv[1] if len(sys.argv) > 1 else "I think this is a really interesting question worth exploring further."
    print_explanation(text)
