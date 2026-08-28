# Human vs. AI-Generated Text: Classification, Model Selection, and Explanation

Classifies a passage of text as human-written or AI-generated, chooses which
model to deploy by multi-criteria decision analysis, and explains individual
predictions with LIME.

CSE710 — Advanced Artificial Intelligence.

**Result.** Four models are trained on the HC3 corpus. A fine-tuned DistilBERT
is the most accurate at 0.9783 against a TF-IDF baseline's 0.9117, but costs
61× the latency, 1168× the storage and 208× the explanation time. TOPSIS and
fuzzy TOPSIS both rank it last, and XGBoost is selected. LIME explanations are
validated by a deletion test before any of them are shown.

Full write-up: [`docs/REPORT.md`](docs/REPORT.md).

## Layout

```
.
├── src/                     all Python modules
│   ├── paths.py             filesystem locations, resolved from the repo root
│   ├── data_prep.py         downloads HC3, cleans / balances / splits it
│   ├── metrics.py           shared metrics and bootstrap intervals
│   │
│   ├── train.py             baseline: TF-IDF + logistic regression
│   ├── train_candidates.py  linear SVM and XGBoost
│   ├── train_advanced.py    fine-tunes DistilBERT
│   ├── advanced_model.py    sklearn-compatible wrapper for the transformer
│   │
│   ├── explain.py           LIME for a single prediction
│   ├── validate_lime.py     faithfulness / stability / sharpness checks
│   │
│   ├── fuzzify.py           linguistic scale and rating bands
│   ├── weights.py           criterion importance
│   ├── topsis.py            TOPSIS and fuzzy TOPSIS
│   ├── benchmark.py         measures every criterion for every model
│   ├── select_model.py      ranks the models, records the choice
│   │
│   ├── make_figures.py      report charts
│   ├── demo.py              CLI demo
│   └── compare.py           all four models, side by side
│
├── data/                    dataset.csv, train.csv, test.csv
├── models/                  trained models + selected.json
├── results/                 decision_matrix.json, lime_validation.json
├── docs/                    REPORT.md and figures/
├── explanations/            LIME highlighted-text HTML
└── requirements.txt
```

Scripts resolve their paths from the repository root rather than the working
directory, so `python src/demo.py` and `cd src && python demo.py` behave
identically.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

XGBoost needs the OpenMP runtime, which pip cannot install:

```bash
brew install libomp          # macOS; on most Linux distributions it is already present
```

## Running it

Each step depends on the ones before it.

```bash
python src/data_prep.py         # build the dataset          (~1 min, downloads HC3)

python src/train.py             # baseline                   (seconds)
python src/train_candidates.py  # SVM + XGBoost              (~1 min)
python src/train_advanced.py    # DistilBERT                 (~10 min on Apple M4)

python src/benchmark.py         # measure all five criteria  (~3 min)
python src/select_model.py      # rank and choose a model    (seconds)
python src/validate_lime.py     # check the explanations     (~5 min)
python src/make_figures.py      # report charts              (seconds)

python src/demo.py --html       # explain the selected model
python src/compare.py           # all four models compared
```

`benchmark.py` and `validate_lime.py` are slow because they time LIME itself,
which means running hundreds of explanations.

## What each stage does

**Training.** Three models share an identical TF-IDF front end — word 1–2
grams, 5000 features — so differences between them come from the learning
algorithm rather than the features. The fourth fine-tunes `distilbert-base-uncased`
for three epochs. All four expose the same `predict` / `predict_proba`
interface over raw strings, which is what lets one LIME implementation serve
all of them.

**Selection.** `benchmark.py` measures five criteria per model: accuracy and F1
to be maximised, and inference latency, model size and LIME explanation time to
be minimised. `select_model.py` ranks the models on those criteria with TOPSIS
and fuzzy TOPSIS, and writes the winner to `models/selected.json`, which
`demo.py` then reads.

**Explanation.** `explain.py` wraps LIME. `validate_lime.py` checks the
explanations are faithful before they are trusted: it deletes the words LIME
credits and confirms the model's prediction actually moves.

## Self-tests

Three modules check themselves and exit non-zero on failure:

```bash
python src/topsis.py         # reproduces two published worked examples
python src/fuzzify.py        # scale construction and threshold boundaries
python src/weights.py        # weight derivation and internal consistency
python src/validate_lime.py  # fails if any model's explanations are unfaithful
```

`topsis.py` is the important one. It reproduces the closeness coefficients of
two published worked examples — a crisp car-selection problem and a fuzzy
hiring problem — so the ranking arithmetic is verified against known answers
rather than against itself.

## Reproducing the reported numbers

Every figure in the report is generated by the scripts above; none are drawn by
hand. Randomness is seeded throughout (`random_state=42` for the splits and the
classical models, `seed=42` for the transformer and for LIME's perturbation
sampling), so a clean run reproduces the report exactly, with two caveats:

- Timing criteria (latency, LIME time) depend on the machine. Their *ordering*
  holds anywhere; the ratios were measured on an Apple M4 using the MPS backend.
- `train_advanced.py` on a different accelerator may land a few hundredths from
  0.9783.

## Submitting

`.venv/` (~880 MB) and `models/distilbert/` (~256 MB) should not be included in
an archive — both are reproducible. `.gitignore` already excludes them.

```bash
zip -r human_ai.zip . -x '.venv/*' 'models/distilbert/*' '.cache/*' '**/__pycache__/*'
```

Rebuild the transformer afterwards with `python src/train_advanced.py`.
