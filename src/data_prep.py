"""Builds the Human-vs-AI text dataset from HC3 (Hello-SimpleAI/HC3)."""

import argparse
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from paths import DATASET_CSV, TEST_CSV, TRAIN_CSV, ensure_dirs

SAMPLES_PER_CLASS = 1500

# HC3 replaced links in human answers with URL_n; AI answers have none, so
# the token leaks the label (12.9% of human rows vs 0.1% of AI).
PLACEHOLDER_RE = re.compile(r"\s*\bURL_\d+\b\s*")


def strip_placeholders(text):
    """Remove URL_n placeholders, leaving non-matching text byte-identical."""
    if not PLACEHOLDER_RE.search(text):
        return text
    return PLACEHOLDER_RE.sub(" ", text).strip()


def load_raw():
    """Load HC3 (all sources combined) from Hugging Face."""
    from datasets import load_dataset

    hc3 = load_dataset("Hello-SimpleAI/HC3", revision="refs/convert/parquet")
    return hc3["train"]


def flatten_to_rows(hc3_train):
    """Each HC3 row bundles a question with a list of human answers and a list
    of chatgpt answers."""
    human_texts = []
    ai_texts = []
    for row in hc3_train:
        human_texts.extend(row["human_answers"])
        ai_texts.extend(row["chatgpt_answers"])

    human_df = pd.DataFrame({"text": human_texts, "label": 0})
    ai_df = pd.DataFrame({"text": ai_texts, "label": 1})
    return pd.concat([human_df, ai_df], ignore_index=True)


def clean(df):
    """Strip whitespace and drop empty/null text rows."""
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.dropna(subset=["text", "label"])
    return df


def sample_balanced(df, n_per_class=SAMPLES_PER_CLASS):
    """Randomly sample n_per_class rows from each label, then shuffle."""
    parts = []
    for label, group in df.groupby("label"):
        n = min(n_per_class, len(group))
        parts.append(group.sample(n=n, random_state=42))
    balanced_df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-dataset", action="store_true",
                        help="rebuild the splits from the existing dataset.csv "
                             "instead of re-downloading HC3")
    args = parser.parse_args()

    ensure_dirs()

    if args.from_dataset:
        # Re-downloading risks a different Hub revision reshuffling the
        # sample, which would change which rows land in train vs test.
        df = clean(pd.read_csv(DATASET_CSV))
        print(f"Loaded {len(df)} rows from {DATASET_CSV}")
    else:
        print("Loading HC3 dataset from Hugging Face...")
        df = clean(flatten_to_rows(load_raw()))

        print("Class counts before sampling:")
        print(df["label"].value_counts())

        df = sample_balanced(df)

        print("\nClass counts after sampling:")
        print(df["label"].value_counts())

        df.to_csv(DATASET_CSV, index=False, encoding="utf-8")
        print(f"\nSaved {len(df)} rows to {DATASET_CSV}")

    # Applied after dataset.csv is written so it stays the raw archive.
    scrubbed = df["text"].map(strip_placeholders)
    print(f"Stripped URL_n from {(scrubbed != df['text']).sum()} rows")
    df = df.assign(text=scrubbed)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_df.to_csv(TRAIN_CSV, index=False, encoding="utf-8")
    test_df.to_csv(TEST_CSV, index=False, encoding="utf-8")

    print(f"Train: {len(train_df)} rows -> {TRAIN_CSV}")
    print(f"Test:  {len(test_df)} rows -> {TEST_CSV}")


if __name__ == "__main__":
    main()
