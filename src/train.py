"""
train.py
--------
Trains the Human-vs-AI text classifier (label 0 = human, label 1 = AI).

Pipeline: TF-IDF vectorizer -> Logistic Regression classifier, wrapped
in a single sklearn Pipeline so explain.py/demo.py can call
`pipeline.predict_proba(raw_text)` directly — LIME needs a function
that takes raw strings in and returns class probabilities out, without
us having to separately vectorize text ourselves.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from paths import BASELINE_MODEL, TEST_CSV, TRAIN_CSV, ensure_dirs


def build_pipeline():
    """
    TfidfVectorizer(ngram_range=(1,2)): use both single words and pairs
    of consecutive words as features, since AI-generated text often has
    telltale phrasing (e.g. "as an ai") that shows up in bigrams.
    max_features=5000: caps the vocabulary size, keeping training fast
    and avoiding overfitting to rare words.

    LogisticRegression does binary classification here (human vs AI)
    with no extra config needed.
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def print_top_words_per_class(pipeline, top_n=15):
    """
    Print the top N words most associated with each class, based on
    the logistic regression coefficients. For binary classification,
    sklearn stores a single coefficient row: positive weight = pushes
    toward class 1 (AI), negative weight = pushes toward class 0
    (human). This is a quick sanity check that the model learned
    something sensible.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]

    top_ai_idx = np.argsort(coefs)[::-1][:top_n]
    top_human_idx = np.argsort(coefs)[:top_n]

    print(f"\nTop {top_n} words most associated with AI-generated text:")
    print("  ", ", ".join(feature_names[top_ai_idx]))

    print(f"\nTop {top_n} words most associated with human text:")
    print("  ", ", ".join(feature_names[top_human_idx]))


def main():
    ensure_dirs()
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    pipeline = build_pipeline()
    pipeline.fit(train_df["text"], train_df["label"])

    predictions = pipeline.predict(test_df["text"])

    print(f"Accuracy: {accuracy_score(test_df['label'], predictions):.3f}")
    print("\nPer-class precision/recall/F1:")
    print(classification_report(test_df["label"], predictions, target_names=["human", "ai"]))

    print("Confusion matrix (rows=true, columns=predicted; order = [human, ai]):")
    labels = pipeline.named_steps["clf"].classes_
    cm = confusion_matrix(test_df["label"], predictions, labels=labels)
    cm_df = pd.DataFrame(cm, index=["human", "ai"], columns=["human", "ai"])
    print(cm_df)

    print_top_words_per_class(pipeline)

    joblib.dump(pipeline, BASELINE_MODEL)
    print(f"\nModel saved to {BASELINE_MODEL}")


if __name__ == "__main__":
    main()
