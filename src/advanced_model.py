"""
advanced_model.py
-----------------
Wraps a fine-tuned DistilBERT sequence classifier so it can be used
anywhere the project already expects an sklearn Pipeline.

Why the wrapper exists
----------------------
explain.py drives LIME through exactly two calls on whatever model it is
given: `pipeline.predict(texts)` and `pipeline.predict_proba(texts)`,
both taking a list of raw strings. The sklearn Pipeline in train.py
satisfies that because its TF-IDF step vectorises internally.

A HuggingFace model does not: it needs tokenising, tensor conversion,
a forward pass, and a softmax before it produces anything comparable.
`TransformerPipeline` performs those steps behind the same two method
names, so the entire existing explainability layer works on the
transformer without a single change to explain.py or demo.py. The
duck-typed interface is what keeps the LIME code model-agnostic in
practice, not just in principle.

Why batching matters here
-------------------------
LIME explains one prediction by generating hundreds or thousands of
perturbed copies of the input text and scoring all of them. It hands
that whole batch to `predict_proba` in a single call. Looping over the
texts one at a time would make each explanation take minutes on a
transformer; batching them into groups of 32 keeps it to seconds, which
is the difference between a usable live demo and an unusable one.
"""

import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from paths import DISTILBERT_DIR

MODEL_DIR = DISTILBERT_DIR

# HC3 answers average roughly 865 characters (~200 word-piece tokens),
# so 256 covers the large majority of inputs intact while costing a
# quarter of the compute a 512-token window would.
MAX_LENGTH = 256
BATCH_SIZE = 32


def pick_device():
    """
    Prefer Apple Silicon's Metal backend, fall back to CUDA, then CPU.

    Kept as a module-level function so train_advanced.py and benchmark.py
    report the same device the model actually runs on.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TransformerPipeline:
    """
    Minimal sklearn-Pipeline-compatible interface over a fine-tuned
    transformer: `predict`, `predict_proba`, and `classes_`.
    """

    def __init__(self, model_dir=MODEL_DIR, device=None, batch_size=BATCH_SIZE):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"No fine-tuned model at {model_dir!r}. Run train_advanced.py first."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = device or pick_device()
        self.model.to(self.device)
        # eval() disables dropout; without it, repeated calls on the same
        # text would return different probabilities and LIME's local
        # model would be fitting noise.
        self.model.eval()
        self.batch_size = batch_size

    @property
    def classes_(self):
        """
        Matches the label encoding fixed in data_prep.py: 0 = human,
        1 = AI. explain.py indexes CLASS_NAMES with the predicted label,
        so this ordering must not drift.
        """
        return np.array([0, 1])

    def predict_proba(self, texts):
        """
        Return an (n, 2) array of class probabilities for a list of raw
        strings — the exact signature LIME requires of a classifier_fn.
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(t) for t in texts]

        outputs = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                logits = self.model(**encoded).logits
                # softmax, not raw logits: LIME fits a linear model on
                # these values and demo.py prints them as confidences,
                # so they have to be real probabilities summing to 1.
                probs = torch.softmax(logits, dim=-1)
                outputs.append(probs.cpu().numpy())

        return np.vstack(outputs)

    def predict(self, texts):
        return self.predict_proba(texts).argmax(axis=1)


def load_advanced_model(model_dir=MODEL_DIR):
    """Mirror of explain.py's load_model(), for the transformer."""
    return TransformerPipeline(model_dir)
