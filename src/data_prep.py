"""
data_prep.py
------------
Builds the Human-vs-AI text dataset from HC3 (Hello-SimpleAI/HC3).

Steps, in order:
  1. Download/load the HC3 "all" config via the `datasets` library.
     Each HC3 row has a "human_answers" list and a "chatgpt_answers"
     list (multiple answers per question), so we flatten those out
     into individual (text, label) rows: label 0 = human, label 1 = AI.
  2. Minimal cleaning: strip whitespace, drop empty entries.
  3. Downsample each class to ~1500 examples so training stays fast,
     then shuffle.
  4. Save the full set to data/dataset.csv (columns: text, label), then
     split 80/20 into data/train.csv and data/test.csv, stratified by
     label so both splits keep the same class balance.
"""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from paths import DATASET_CSV, TEST_CSV, TRAIN_CSV, ensure_dirs

SAMPLES_PER_CLASS = 1500


def load_raw():
    """
    Load HC3 (all sources combined) from Hugging Face.

    HC3's original loading script is no longer supported by newer
    versions of the `datasets` library, so we load from the Hub's
    auto-generated Parquet mirror instead (revision="refs/convert/parquet").
    That mirror only exposes a single "default" config, which already
    contains every source combined — equivalent to the original "all"
    config — so no config name is passed here.
    """
    hc3 = load_dataset("Hello-SimpleAI/HC3", revision="refs/convert/parquet")
    return hc3["train"]


def flatten_to_rows(hc3_train):
    """
    Each HC3 row bundles a question with a list of human answers and a
    list of chatgpt answers. Flatten every individual answer into its
    own (text, label) row: label 0 = human, label 1 = chatgpt/AI.
    """
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
    print("Loading HC3 dataset from Hugging Face...")
    hc3_train = load_raw()

    df = flatten_to_rows(hc3_train)
    df = clean(df)

    print("Class counts before sampling:")
    print(df["label"].value_counts())

    df = sample_balanced(df)

    print("\nClass counts after sampling:")
    print(df["label"].value_counts())

    ensure_dirs()
    df.to_csv(DATASET_CSV, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} rows to {DATASET_CSV}")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_df.to_csv(TRAIN_CSV, index=False, encoding="utf-8")
    test_df.to_csv(TEST_CSV, index=False, encoding="utf-8")

    print(f"Train: {len(train_df)} rows -> {TRAIN_CSV}")
    print(f"Test:  {len(test_df)} rows -> {TEST_CSV}")


if __name__ == "__main__":
    main()
