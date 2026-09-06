"""Fine-tunes DistilBERT on the Human-vs-AI task (0 = human, 1 = AI)."""

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from advanced_model import MAX_LENGTH, TransformerPipeline, pick_device
from metrics import confusion, evaluate

from paths import DISTILBERT_DIR, TEST_CSV, TRAINER_CACHE, TRAIN_CSV, ensure_dirs

BASE_MODEL = "distilbert-base-uncased"
SEED = 42


class TextDataset(torch.utils.data.Dataset):
    """Wraps pre-tokenised encodings plus labels in the item-dict format
    Trainer expects."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def main():
    set_seed(SEED)
    device = pick_device()
    print(f"Device: {device}")

    ensure_dirs()
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2
    )

    def encode(texts):
        return tokenizer(
            [str(t) for t in texts],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

    train_ds = TextDataset(encode(train_df["text"]), train_df["label"])

    args = TrainingArguments(
        output_dir=str(TRAINER_CACHE),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        seed=SEED,
        logging_steps=50,
        save_strategy="no",     # only the final model is needed
        report_to="none",       # no wandb/tensorboard side effects
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds)

    print("\nFine-tuning DistilBERT...")
    trainer.train()

    model.save_pretrained(DISTILBERT_DIR)
    tokenizer.save_pretrained(DISTILBERT_DIR)
    print(f"\nModel saved to {DISTILBERT_DIR}")

    # Evaluate through the same wrapper the demo and LIME will use.
    print("\nEvaluating...")
    pipeline = TransformerPipeline(DISTILBERT_DIR)
    predictions = pipeline.predict(test_df["text"].tolist())
    y_true = test_df["label"].values

    scores = evaluate(y_true, predictions)
    print(f"\nAccuracy:    {scores['accuracy']:.4f}")
    print(f"F1:          {scores['f1']:.4f}")
    print(f"Sensitivity: {scores['sensitivity']:.4f}  (recall of AI class)")
    print(f"Specificity: {scores['specificity']:.4f}  (recall of human class)")

    print("\nConfusion matrix (rows=true, columns=predicted; order = [human, ai]):")
    cm = pd.DataFrame(
        confusion(y_true, predictions),
        index=["human", "ai"],
        columns=["human", "ai"],
    )
    print(cm)


if __name__ == "__main__":
    main()
