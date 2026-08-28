"""
train_candidates.py
-------------------
Trains the two additional classical candidates — Linear SVM and
XGBoost — so the multi-criteria ranking in select_model.py has a
spread of alternatives rather than just "cheap linear model vs.
expensive transformer".

Both reuse the TF-IDF front end from train.py (word 1-2 grams, top 5000
features) and differ only in the classifier, which keeps the comparison
honest: any difference in score comes from the learning algorithm, not
from different text features.

Why the SVM is wrapped in CalibratedClassifierCV
------------------------------------------------
LinearSVC has no predict_proba. It only exposes decision_function, a
signed distance from the separating hyperplane, which is unbounded and
not a probability. That is a problem here for two reasons: LIME requires
a classifier_fn returning per-class probabilities, and demo.py prints a
confidence percentage.

CalibratedClassifierCV fits the SVM inside a cross-validation loop and
then fits a calibration curve (Platt scaling by default) mapping those
distances onto probabilities. The result exposes predict_proba and slots
into the same interface as every other model in the project.
"""

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from metrics import confusion, evaluate

from paths import SVM_MODEL, TEST_CSV, TRAIN_CSV, XGBOOST_MODEL, ensure_dirs

SEED = 42


def build_vectorizer():
    """Identical settings to train.py's pipeline, so features match."""
    return TfidfVectorizer(ngram_range=(1, 2), max_features=5000)


def build_svm():
    return Pipeline([
        ("tfidf", build_vectorizer()),
        # cv=5: five folds of calibration data. dual="auto" silences the
        # solver-choice warning on sklearn >= 1.3 for this n_samples /
        # n_features ratio.
        ("clf", CalibratedClassifierCV(
            LinearSVC(dual="auto", random_state=SEED), cv=5
        )),
    ])


def build_xgboost():
    return Pipeline([
        ("tfidf", build_vectorizer()),
        # Modest depth and a few hundred trees: TF-IDF features are
        # sparse and high-dimensional, so deep trees overfit quickly on
        # 4.7k rows.
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


if __name__ == "__main__":
    main()
