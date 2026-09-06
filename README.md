# Human vs. AI-Generated Text: Classification, Model Selection, and Explanation

Classifies a passage of text as human-written or AI-generated, chooses which
model to deploy by multi-criteria decision analysis, and explains individual
predictions with LIME.

CSE710 — Advanced Artificial Intelligence.

**Result.** Four models are trained on the HC3 corpus and ranked on six
criteria. A fine-tuned DistilBERT is the most accurate at 0.9817 against a
TF-IDF baseline's 0.9267, but costs 55× the latency, 285× the storage and 137×
the explanation time. TOPSIS and fuzzy TOPSIS both rank it last, and LinearSVM
is selected. LIME explanations are validated by a deletion test before any of
them are shown.

Explanation quality is itself one of the six criteria: `content_share` measures
how much of an explanation is carried by content words rather than `the` and
`and`. A fifth model trained without function words prices that tradeoff — it
costs 4.0 accuracy points and takes content share from 28.4% to 99.9%.

Full write-up: [`docs/REPORT.md`](docs/REPORT.md).
Per-file reference: [`docs/FILES.md`](docs/FILES.md).

## Layout

```
  data/dataset.csv  (raw archive, HC3)
          |
          v
  data_prep.py  --from-dataset      strips URL_n placeholders
          |
          v
  data/train.csv (2400)   data/test.csv (600)
          |
          +---------------+------------------+------------------+
          v               v                  v                  v
      train.py     train_candidates.py                  train_advanced.py
      LogReg       LinearSVM, XGBoost,                   DistilBERT
                   LogReg-Content (ablation)
          |               |                  |                  |
          +---------------+------------------+------------------+
                          v
                   benchmark.py        6 criteria x 5 models
                          v
                results/decision_matrix.json
                          v
                  select_model.py      TOPSIS + fuzzy TOPSIS
                          v            (ablations excluded)
                 models/selected.json  ->  LinearSVM
                          |
          +---------------+------------------+
          v                                  v
   explain.py                         validate_lime.py
   LIME on the selected model         faithfulness, stability
```

```
.
├── src/            all Python modules — see docs/FILES.md
├── data/           dataset.csv, train.csv, test.csv
├── models/         trained models + selected.json
├── results/        decision_matrix.json, lime_validation.json
├── docs/           REPORT.md, FILES.md, figures/
├── explanations/   LIME highlighted-text HTML
└── requirements.txt
```

Scripts resolve their paths from the repository root rather than the working
directory, so `python src/explain.py` and `cd src && python explain.py` behave
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
python src/data_prep.py --from-dataset   # rebuild splits from dataset.csv (seconds)
# omit the flag to re-download HC3 (~1 min); the flag keeps the split identical

python src/train.py             # baseline                   (seconds)
python src/train_candidates.py  # SVM, XGBoost, Content      (~1 min)
python src/train_advanced.py    # DistilBERT                 (~7 min on Apple M4)

python src/benchmark.py         # measure all six criteria   (~4 min)
python src/select_model.py      # rank and choose a model    (seconds)
python src/validate_lime.py     # check the explanations     (~5 min)
python src/make_figures.py      # report charts              (seconds)

python src/explain.py "some text"        # explain one passage
python src/explain.py --html             # ...and write the HTML report
python src/explain.py data/sample_text.txt --model content
python src/explain.py --demo             # the built-in examples, + an HTML report
python src/compare.py                    # every model compared
```

`benchmark.py` and `validate_lime.py` are slow because they time LIME itself,
which means running hundreds of explanations.

`explain.py` uses whichever model `select_model.py` chose. Pass
`--model baseline|svm|xgboost|content|advanced` to override.

## What each stage does

See [`docs/FILES.md`](docs/FILES.md) for a per-file reference.

**Training.** Four models share an identical TF-IDF front end — word 1–2 grams,
20000 features, sublinear term frequency — so differences between them come from
the learning algorithm rather than the features. A fifth fine-tunes
`distilbert-base-uncased` for three epochs. All expose the same `predict` /
`predict_proba` interface over raw strings, which is what lets one LIME
implementation serve all of them.

**Selection.** `benchmark.py` measures six criteria per model: accuracy, F1 and
explanation content share to be maximised, and inference latency, model size and
LIME explanation time to be minimised. `select_model.py` ranks the candidates
with TOPSIS and fuzzy TOPSIS and writes the winner to `models/selected.json`.
Ablations are measured but excluded from the ranking.

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
  0.9817.

Reported test-set results:

| model | accuracy | F1 | content share |
|---|---|---|---|
| DistilBERT | 0.9817 | 0.9820 | 44.6% |
| **LinearSVM** (selected) | 0.9517 | 0.9521 | 32.5% |
| LogisticRegression | 0.9267 | 0.9254 | 28.4% |
| XGBoost | 0.9267 | 0.9236 | 17.1% |
| *LogReg-Content* (ablation) | *0.8867* | *0.8863* | *99.9%* |

## Submitting

`.venv/` (~880 MB) and `models/distilbert/` (~256 MB) should not be included in
an archive — both are reproducible. `.gitignore` already excludes them.

```bash
zip -r human_ai.zip . -x '.venv/*' 'models/distilbert/*' '.cache/*' '**/__pycache__/*'
```

Rebuild the transformer afterwards with `python src/train_advanced.py`.
