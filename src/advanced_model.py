"""Wraps a fine-tuned DistilBERT sequence classifier so it can be used anywhere
the project already expects an sklearn Pipeline."""

import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from paths import DISTILBERT_DIR

MODEL_DIR = DISTILBERT_DIR

# HC3 answers average roughly 865 characters (~200 word-piece tokens).
MAX_LENGTH = 256
BATCH_SIZE = 32


def pick_device():
    """Prefer Apple Silicon's Metal backend, fall back to CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TransformerPipeline:
    """Minimal sklearn-Pipeline-compatible interface over a fine-tuned
    transformer: `predict`, `predict_proba`, and `classes_`."""

    def __init__(self, model_dir=MODEL_DIR, device=None, batch_size=BATCH_SIZE):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"No fine-tuned model at {model_dir!r}. Run train_advanced.py first."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = device or pick_device()
        self.model.to(self.device)
        # eval() disables dropout; without it.
        self.model.eval()
        self.batch_size = batch_size

    @property
    def classes_(self):
        """Matches the label encoding fixed in data_prep.py: 0 = human, 1 = AI."""
        return np.array([0, 1])

    def predict_proba(self, texts):
        """Return an (n, 2) array of class probabilities for a list of raw
        strings — the exact signature LIME requires of a classifier_fn."""
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
                # softmax.
                probs = torch.softmax(logits, dim=-1)
                outputs.append(probs.cpu().numpy())

        return np.vstack(outputs)

    def predict(self, texts):
        return self.predict_proba(texts).argmax(axis=1)


def load_advanced_model(model_dir=MODEL_DIR):
    """Mirror of explain.py's load_model(), for the transformer."""
    return TransformerPipeline(model_dir)
