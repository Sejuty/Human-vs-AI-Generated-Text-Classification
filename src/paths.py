"""Every filesystem location the project uses, resolved from this file rather
than from the working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
EXPLANATIONS_DIR = ROOT / "explanations"

# Datasets
DATASET_CSV = DATA_DIR / "dataset.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

# Trained models
BASELINE_MODEL = MODELS_DIR / "classifier.joblib"
SVM_MODEL = MODELS_DIR / "svm.joblib"
XGBOOST_MODEL = MODELS_DIR / "xgboost.joblib"
CONTENT_MODEL = MODELS_DIR / "content.joblib"
DISTILBERT_DIR = MODELS_DIR / "distilbert"
SELECTED_JSON = MODELS_DIR / "selected.json"

# Measurements and analysis
DECISION_MATRIX = RESULTS_DIR / "decision_matrix.json"
LIME_VALIDATION = RESULTS_DIR / "lime_validation.json"

# Transient scratch space for the transformer trainer.
TRAINER_CACHE = ROOT / ".cache" / "trainer"


def ensure_dirs():
    """Create the output directories the scripts write into."""
    for directory in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR,
                      EXPLANATIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
