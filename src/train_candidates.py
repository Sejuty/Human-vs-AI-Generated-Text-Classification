"""Trains the two additional classical candidates."""

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from metrics import confusion, evaluate
from train import build_vectorizer

from paths import (CONTENT_MODEL, SVM_MODEL, TEST_CSV, TRAIN_CSV,
                   XGBOOST_MODEL, ensure_dirs)

SEED = 42


def build_svm():
    return Pipeline([
        ("tfidf", build_vectorizer()),
        # cv=5: five folds of calibration data.
        ("clf", CalibratedClassifierCV(
            LinearSVC(dual="auto", random_state=SEED), cv=5
        )),
    ])


def build_xgboost():
    return Pipeline([
        ("tfidf", build_vectorizer()),
        # Modest depth and a few hundred trees: TF-IDF features are sparse and
        # high-dimensional, so deep trees overfit quickly on 4.7k rows.
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            n_jobs=-1,
            eval_metric="logloss",
        )),
    ])


def build_content():
    """Function words are excluded, so LIME can only attribute to content
    words. Trades accuracy for explanations a reader can act on."""
    return Pipeline([
        ("tfidf", build_vectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train_and_report(name, pipeline, train_df, test_df, out_path):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    pipeline.fit(train_df["text"], train_df["label"])
    predictions = pipeline.predict(test_df["text"])
    y_true = test_df["label"].values

    scores = evaluate(y_true, predictions)
    print(f"Accuracy:    {scores['accuracy']:.4f}")
    print(f"F1:          {scores['f1']:.4f}")
    print(f"Sensitivity: {scores['sensitivity']:.4f}")
    print(f"Specificity: {scores['specificity']:.4f}")

    print("\nConfusion matrix (rows=true, cols=predicted; [human, ai]):")
    print(pd.DataFrame(
        confusion(y_true, predictions), index=["human", "ai"], columns=["human", "ai"]
    ))

    # Fail loudly rather than at LIME time if probabilities are missing.
    assert hasattr(pipeline, "predict_proba"), f"{name} exposes no predict_proba"

    joblib.dump(pipeline, out_path)
    print(f"\nSaved to {out_path}")


def main():
    ensure_dirs()
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    train_and_report("Linear SVM (calibrated)", build_svm(),
                     train_df, test_df, SVM_MODEL)
    train_and_report("XGBoost", build_xgboost(),
                     train_df, test_df, XGBOOST_MODEL)
    train_and_report("LogisticRegression-Content", build_content(),
                     train_df, test_df, CONTENT_MODEL)


if __name__ == "__main__":
    main()
