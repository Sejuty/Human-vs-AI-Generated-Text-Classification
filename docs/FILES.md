# What each file does

## Pipeline

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
                   models/*.joblib , models/distilbert/
                          |
                          v
                   benchmark.py        6 criteria x 5 models
                          |
                          v
                results/decision_matrix.json
                          |
                          v
                  select_model.py      TOPSIS + fuzzy TOPSIS
                          |            (ablations excluded)
                          v
                 models/selected.json  ->  LinearSVM
                          |
          +---------------+------------------+
          v                                  v
   explain.py                         validate_lime.py
   LIME on the selected model         faithfulness, stability
          |                                  |
          v                                  v
   explanations/*.html            results/lime_validation.json
```

## Stage scripts

Run these in the order above.

| file | what it does |
|---|---|
| `data_prep.py` | Builds the dataset from HC3, strips `URL_n` placeholders, writes the 80/20 split. `--from-dataset` rebuilds from `dataset.csv` without re-downloading. |
| `train.py` | Trains the LogisticRegression baseline. Owns `build_vectorizer()`, the TF-IDF front end every classical model shares. |
| `train_candidates.py` | Trains LinearSVM, XGBoost, and the LogReg-Content ablation. |
| `train_advanced.py` | Fine-tunes DistilBERT (3 epochs). |
| `benchmark.py` | Measures all six criteria for every model; writes `decision_matrix.json`. Marks ablations. |
| `select_model.py` | Ranks candidates by TOPSIS and fuzzy TOPSIS, runs the robustness and sensitivity checks, writes `selected.json`. |
| `validate_lime.py` | Tests that LIME explanations describe the models: deletion test, seed stability, sharpness. |
| `make_figures.py` | Regenerates the figures in `docs/figures/`. |

## Entry points

| file | what it does |
|---|---|
| `explain.py` | Explains one text with LIME using the selected model. `--model` overrides; accepts raw text or a file path. `--demo` runs the built-in example set instead and writes a highlighted HTML report; `--html` writes that report for any input, to `explanations/` unless given a path. Owns `EXAMPLES`, the demonstration set `compare.py` also uses. |
| `compare.py` | Runs every model over the same examples and reports where they disagree. |

## Supporting modules

| file | what it does |
|---|---|
| `paths.py` | Every filesystem location, resolved from the repo root rather than the working directory. |
| `metrics.py` | Accuracy, F1, sensitivity, specificity, confusion matrix, bootstrap CIs, table formatting. |
| `topsis.py` | TOPSIS and fuzzy TOPSIS in plain numpy, with self-tests against published worked examples. |
| `fuzzify.py` | Maps raw measurements onto the linguistic scale and its triangular fuzzy numbers. |
| `weights.py` | Criterion importance, recorded as a linguistic judgement by the decision maker. |
| `advanced_model.py` | `TransformerPipeline`, giving DistilBERT the same `predict_proba` interface as the sklearn models. |

## The six criteria

Measured by `benchmark.py`, ranked by `select_model.py`.

| criterion | direction | importance |
|---|---|---|
| `accuracy` | benefit | very high |
| `f1` | benefit | high |
| `content_share` | benefit | high |
| `latency_ms` | cost | medium |
| `lime_seconds` | cost | medium |
| `size_mb` | cost | low |

`content_share` is the percentage of LIME explanation weight carried by words
that are not English stop words — how much of an explanation a reader can act on.

## Ablation

`LogisticRegression-Content` is trained with `stop_words="english"`, so it cannot
attribute to function words. It is measured and reported but **excluded from the
ranking**: it scores ~100% on `content_share` by construction, so ranking it
would measure how it was built rather than how good it is. It exists to price the
tradeoff — against the otherwise identical LogReg baseline it costs 4.0 accuracy
points (0.9267 → 0.8867) and moves content share from 28.4% to 99.9%.
