# Human vs. AI-Generated Text: Classification, Model Selection, and Explanation

**CSE710 — Advanced Artificial Intelligence**

A binary text classifier distinguishing human-written from AI-generated writing,
in which the deployed model is chosen by multi-criteria decision analysis and
its individual predictions are explained with LIME.

Every figure in this report is produced by the scripts in this repository.

---

## 1. Problem and dataset

The task is to classify a passage of text as human-written (label 0) or
AI-generated (label 1).

**Source.** HC3, the Human-ChatGPT Comparison Corpus (Guo et al., 2023),
obtained from the Hugging Face distribution. Each HC3 record pairs a question
with a list of human answers and a list of ChatGPT answers.

**Construction** (`src/data_prep.py`):

1. **Flatten** — every individual answer becomes its own row, labelled by
   origin. Answers are never concatenated, so each row is one author's complete
   response to one question.
2. **Clean** — strip whitespace, drop empty and null rows.
3. **Balance** — sample 1,500 rows per class, then shuffle. A balanced set makes
   accuracy directly interpretable and removes any majority-class prior.
4. **Split** — 80/20 stratified, `random_state=42`.

| | rows |
|---|---|
| Full dataset | 3,000 (1,500 human / 1,500 AI) |
| Train | 2,400 |
| Test | 600 (300 human / 300 AI) |

Mean text length is roughly 865 characters. These are long-form answers rather
than single sentences, which matters for interpreting the out-of-distribution
behaviour discussed in §5.6.

**A caveat, stated up front.** The pipeline does not deduplicate, so
near-duplicate HC3 answers may fall on both sides of the split. This inflates
every model's score by a similar amount, leaving the comparison and the ranking
fair, but the absolute accuracies below should be read as optimistic.

---

## 2. Candidate models

Four models, all trained and evaluated on the identical split.

| Model | Representation | Classifier |
|---|---|---|
| Logistic Regression | TF-IDF, 1–2 grams, 5,000 features | linear, `max_iter=1000` |
| Linear SVM | same TF-IDF | `LinearSVC` + `CalibratedClassifierCV(cv=5)` |
| XGBoost | same TF-IDF | 300 trees, depth 6, learning rate 0.1 |
| DistilBERT | learned word-piece embeddings | fine-tuned transformer |

The three classical models share an identical TF-IDF front end, so any
difference between them arises from the learning algorithm rather than from
different features.

**The transformer.** DistilBERT (Sanh et al., 2019) is a distilled six-layer,
~66M-parameter model. Two properties separate it categorically from the
bag-of-words classifiers. *Self-attention* builds each token's representation
from the entire surrounding passage, so a phrase is read in context rather than
as independent unigrams; the TF-IDF models cannot see word order at all.
*Transfer learning* means the model arrives already knowing English from
large-scale pretraining, and fine-tuning only adapts it — which is why 2,400
training rows suffice, where a network of this size trained from scratch on that
data would not converge.

Fine-tuning: 3 epochs, learning rate 2e-5, batch size 16, `seed=42`, maximum
sequence length 256 tokens. Training took 10 min 04 s on an Apple M4 using the
MPS backend, with training loss falling from approximately 0.5 to 0.0043.

**A note on the SVM.** `LinearSVC` exposes only `decision_function`, an unbounded
signed distance from the separating hyperplane, not probabilities. LIME requires
a function returning class probabilities, and the demo prints a confidence, so
the SVM is wrapped in `CalibratedClassifierCV`, which fits a calibration curve
mapping distances onto probabilities.

---

## 3. Classification results

All figures on the 600-row held-out test set.

| Model | Accuracy | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| Logistic Regression | 0.9117 | 0.9112 | 0.9067 | 0.9167 |
| Linear SVM | 0.9267 | 0.9272 | 0.9333 | 0.9200 |
| XGBoost | 0.9367 | 0.9354 | 0.9167 | 0.9567 |
| **DistilBERT** | **0.9783** | **0.9788** | **1.0000** | 0.9567 |

Sensitivity is recall of the AI class; specificity is recall of the human class.

![Accuracy with confidence intervals](figures/quality_intervals.png)

**95% bootstrap confidence intervals** (1,000 resamples of the test set):

| Model | Accuracy 95% CI |
|---|---|
| Logistic Regression | [0.8883, 0.9350] |
| Linear SVM | [0.9050, 0.9483] |
| XGBoost | [0.9183, 0.9567] |
| DistilBERT | [0.9667, 0.9900] |

The intervals decide which gaps are real. Logistic Regression, Linear SVM and
XGBoost overlap substantially with one another, so their ordering is not
meaningful on a test set this size. DistilBERT's interval is disjoint from all
three, so its advantage is genuine rather than sampling noise.

![Confusion matrices](figures/confusion_matrices.png)

DistilBERT's error profile is asymmetric and worth noting: it missed **none** of
the 300 AI texts, and all 13 of its errors were human writing filed as AI. In a
plagiarism-detection setting that is the more damaging direction — a false
accusation — even though it produces the best headline accuracy.

---

## 4. Multi-criteria model selection

### 4.1 Why accuracy alone cannot decide

An obvious approach is to rank the models on Accuracy, F1, Sensitivity and
Specificity. On this dataset that would be a **degenerate** multi-criteria
problem. The test set is exactly balanced at 300/300, so

```
Accuracy = (Sensitivity + Specificity) / 2
```

holds identically, not approximately — confirmed on the baseline:
(0.9067 + 0.9167) / 2 = 0.9117. F1 tracks the same underlying quantities. Those
four criteria are one dimension wearing four hats, and ranking on them
reproduces the accuracy ordering with extra arithmetic.

The criteria that genuinely conflict are the ones a results table usually omits
(`src/benchmark.py`):

| Model | Accuracy | F1 | Latency (ms) | Size (MB) | LIME time (s) |
|---|---|---|---|---|---|
| Logistic Regression | 0.9117 | 0.9112 | 0.285 | 0.22 | 0.051 |
| Linear SVM | 0.9267 | 0.9272 | 1.067 | 0.37 | 0.053 |
| XGBoost | 0.9367 | 0.9354 | 0.370 | 0.65 | 0.053 |
| DistilBERT | 0.9783 | 0.9788 | 17.272 | 256.33 | 10.639 |

DistilBERT is the most accurate model and simultaneously **61× slower**,
**1,168× larger** and **208× more expensive to explain**. No model is best on
everything, which is the precondition for a multi-criteria method to be doing
real work.

![Decision criteria](figures/criteria_comparison.png)

The figure shows the shape of the decision at a glance: the two quality panels
are nearly flat across models, while the three cost panels — all log-scaled —
have DistilBERT towering over the rest.

Latency is measured one text at a time rather than batched, because the demo and
the explainer classify single inputs; that is the latency a user experiences.
Including **LIME explanation time** as a criterion is what ties the two halves of
this project together: in a system whose purpose is explainable classification,
the cost of producing an explanation is a property of the model, not an
implementation detail.

### 4.2 Criterion importance

TOPSIS does not derive weights; they are supplied by the decision-maker
(Hwang & Yoon, 1981). That makes the weighting the one genuinely subjective step,
so it is recorded as a linguistic judgement (`src/weights.py`) rather than as
bare decimals:

| Criterion | Importance | TFN | Crisp weight | Rationale |
|---|---|---|---|---|
| accuracy | Very High | (7, 9, 9) | 0.294 | the headline measure of the task |
| f1 | High | (5, 7, 9) | 0.247 | balances the two error directions |
| latency_ms | Medium | (3, 5, 7) | 0.176 | experienced on every use |
| size_mb | Low | (1, 3, 5) | 0.106 | nothing here targets a constrained device |
| lime_seconds | Medium | (3, 5, 7) | 0.176 | waiting for the explanation costs as much as waiting for the prediction |

Fuzzy TOPSIS consumes the TFNs directly; crisp TOPSIS uses their centroids
`(l+m+u)/3`, normalised. Deriving both from a single judgement prevents the two
from drifting apart.

**How much the weighting matters: not much.** Re-ranking under three different
schemes leaves the conclusion intact:

| Weighting | TOPSIS | Fuzzy TOPSIS |
|---|---|---|
| Uniform | XGBoost > LogReg > SVM > DistilBERT | LogReg = XGBoost > SVM > DistilBERT |
| Linguistic (adopted) | XGBoost > SVM > LogReg > DistilBERT | LogReg = XGBoost > SVM > DistilBERT |
| Quality-led | XGBoost > SVM > LogReg > DistilBERT | LogReg = XGBoost > SVM > DistilBERT |

XGBoost ranks first and DistilBERT last under every scheme; only second and third
place exchange. Demonstrating that the result survives the weighting is a
stronger defence than any argument for one particular set of numbers.

### 4.3 TOPSIS

Vector normalisation, weighting, ideal and anti-ideal solutions, Euclidean
distances, closeness `CC = d⁻/(d⁺+d⁻)`.

| Rank | Model | d⁺ | d⁻ | CC |
|---|---|---|---|---|
| 1 | **XGBoost** | 0.0087 | 0.2678 | **0.9685** |
| 2 | Linear SVM | 0.0132 | 0.2633 | 0.9521 |
| 3 | Logistic Regression | 0.0137 | 0.2684 | 0.9514 |
| 4 | DistilBERT | 0.2684 | 0.0137 | 0.0486 |

### 4.4 Fuzzy TOPSIS

Ratings are assigned by comparing each measurement against threshold bands and
looking up the resulting category's triangular fuzzy number (Zadeh, 1965;
Chen, 2000). The scale is built by construction — six categories spread over
0–10, peaks two apart, each triangle spanning one step either side, ends clipped
— giving `poor (0,0,2)` through `outstanding (8,10,10)`. The triangles overlap so
that a value on a band boundary shades between categories rather than jumping.

Accuracy and F1 use conventional percentage bands (>95 outstanding, >90
excellent, >80 very good, …). Latency, size and explanation time are not
percentages, so their bands are a stated assumption recorded in full in
`src/fuzzify.py` — absolute judgements about what each quantity means for an
interactively driven classifier, deliberately not derived from the observed
measurements, so they do not shift when the candidate set changes.

![Linguistic decision matrix](figures/linguistic_matrix.png)

| Model | accuracy | f1 | latency | size | LIME time |
|---|---|---|---|---|---|
| Logistic Regression | excellent | excellent | outstanding | outstanding | outstanding |
| Linear SVM | excellent | excellent | excellent | outstanding | outstanding |
| XGBoost | excellent | excellent | outstanding | outstanding | outstanding |
| DistilBERT | outstanding | outstanding | very good | good | good |

Distances use the vertex metric, and the fuzzy ideal and anti-ideal are computed
as the componentwise maximum and minimum of the weighted matrix per criterion.

| Rank | Model | CC |
|---|---|---|
| 1= | **Logistic Regression** | **0.7134** |
| 1= | **XGBoost** | **0.7134** |
| 3 | Linear SVM | 0.6368 |
| 4 | DistilBERT | 0.2866 |

**The tie is real and is reported rather than broken silently.** Logistic
Regression and XGBoost hold identical categories on all five criteria: 0.9117 and
0.9367 are both simply ">90% excellent". The linguistic scale is too coarse to
separate them, which is a substantive statement about how similar they are, not a
defect. TOPSIS, working from the raw values, does separate them.

### 4.5 What the two methods agree on

Both rank **DistilBERT last**. Its accuracy advantage is worth one linguistic
category — excellent to outstanding — while its cost drops it two categories on
both size and explanation time. Selection follows the crisp ranking, on the
stated grounds that it works from raw values and can distinguish models the
linguistic scale rates identically.

**Selected model: XGBoost.**

This is the substantive finding of the analysis. A measurably better classifier
exists, and the decision analysis concludes that its cost is not justified in
this setting. That is precisely what a multi-criteria method is for: it makes the
trade-off explicit and auditable instead of leaving it to whoever reads the
accuracy column first.

### 4.6 Sensitivity to the quality/cost balance

Sweeping the weight given to quality criteria from 0 to 1, with cost taking the
remainder:

![Weight sensitivity](figures/sensitivity.png)

| Quality weight | TOPSIS winner |
|---|---|
| 0.00 – 0.05 | Logistic Regression |
| 0.10 – 0.95 | **XGBoost** |
| 1.00 | DistilBERT |

DistilBERT wins only at a quality weight of exactly 1.00 — that is, only when
cost is given no weight whatsoever. XGBoost holds the entire practical range.
This is not a robustness check that happened to pass; it locates the precise
condition under which the recommendation would change, and that condition is
degenerate.

### 4.7 What the ranking is not used for

It selects one model. It does not weight an ensemble. Closeness coefficients rank
alternatives; they are not calibrated probabilities, and once cost criteria are
in the matrix, using them as vote weights would mean treating a model's
cheapness as a reason to trust its verdict.

---

## 5. Explainability

### 5.1 Method

LIME (Ribeiro et al., 2016) explains a single prediction by perturbing the input
— removing words — observing how the output probabilities shift, and fitting a
small local linear model to that behaviour. The linear model's weights become the
word-level importance scores. It is a *local* approximation: an explanation of
one prediction, not a summary of the model.

Scores throughout this report are expressed relative to the AI class, so a
positive weight means the word pushes toward "AI-generated" and a negative one
toward "human", regardless of which class was predicted.

### 5.2 Why a model-agnostic explainer is necessary

For the logistic regression baseline, LIME is arguably redundant: that model's
coefficients can be read off directly, as `src/train.py` does when it prints the
top 15 words per class. Explaining a linear model with a local linear
approximation is close to circular.

For DistilBERT there is no such shortcut. Importance is distributed across 66M
parameters and six layers of attention; no single weight corresponds to a word.
The only way to learn what the model responds to is to probe it from outside —
change the input, observe the output. That is exactly what LIME does, and it is
why the absence of a coefficient-inspection routine for the transformer is not an
omission but the reason the project needs a model-agnostic explainer.

### 5.3 Validating the explanations

An explanation is persuasive by construction: highlighted words read as a reason
whether or not the model used them. Before any explanation is presented, three
properties are measured (`src/validate_lime.py`).

**Faithfulness — the deletion test.** Take the words LIME says support the
predicted class, remove them, and re-classify. If those words are what the model
was using, the predicted-class probability should fall substantially.

**Stability.** LIME samples perturbations, so its output is stochastic. Re-running
each explanation under several seeds and correlating the word weights measures
that variance directly.

**Sharpness.** The magnitude of the largest weight, reported alongside model
confidence.

| Model | Faithfulness (Δ probability) | Sharpness | Stability | Mean confidence |
|---|---|---|---|---|
| Logistic Regression | +0.2321 | 0.1200 | 0.997 | 0.693 |
| Linear SVM | +0.4754 | 0.2301 | 0.993 | 0.912 |
| **XGBoost** | **+0.2508** | 0.1325 | **0.925** | 0.987 |
| DistilBERT | +0.1736 | 0.2389 | 0.557 | 0.999 |

Every model shows a clearly positive probability drop, so LIME is identifying
words the models genuinely use rather than producing plausible noise. The
selected model, XGBoost, is faithful (+0.2508) and stable (0.925).

**Stability falls as confidence rises.** DistilBERT, at 0.999 mean confidence,
returns explanations that correlate only 0.557 across perturbation seeds — its
word weights move substantially between runs. This corroborates the ranking of
§4 from an independent direction: the transformer is not merely expensive to
explain, its explanations are also the least reproducible.

### 5.4 Where the models disagree

Of eleven demonstration examples, two split the models (`src/compare.py`). The
clearest is:

> *"I appreciate your question and would be happy to help clarify this for you."*

| Model | Prediction | Confidence | Top LIME words |
|---|---|---|---|
| Logistic Regression | Human | 68.2% | would −0.18, help +0.14, and +0.09 |
| Linear SVM | Human | 76.0% | would −0.31, help +0.26, and +0.15 |
| XGBoost | Human | 96.5% | help +0.06, would −0.04, this −0.03 |
| **DistilBERT** | **AI-generated** | **95.2%** | **appreciate +0.29, your +0.19, to +0.13** |

The three bag-of-words models fixate on common function words. DistilBERT keys on
*"appreciate"* and *"your"* — the register of assistant writing. Comparing the two
models on the same two words is the sharpest illustration in the project:

| Word | Logistic Regression | DistilBERT |
|---|---|---|
| appreciate | −0.005 | **+0.286** |
| your | +0.019 | **+0.182** |

For the bag-of-words model these words are statistically inert — weights
indistinguishable from noise, because "appreciate" is not frequent enough in HC3
for TF-IDF to have learned anything about it. For DistilBERT they are the two
strongest pieces of evidence in the sentence. That is contextual representation
doing something a term frequency cannot, and LIME is what makes it visible rather
than a matter of trusting the more accurate model.

The second disagreement is an HC3 passage of encyclopedic biography whose true
label is AI. The three bag-of-words models and the transformer split on it, with
the selected model on the losing side — recorded here because a demonstration set
that only contains cases the chosen model gets right is not evidence of anything.

### 5.5 Confidence and explainability pull against each other

The sharpness column of §5.3 reflects a structural property, not a defect. When a
model outputs 99.9% confidence, removing any single word leaves the probability
essentially unchanged, so the local linear model LIME fits has almost no gradient
to report. The better separated the decision boundary, the less informative the
local explanation becomes. A system that wants both must accept a trade-off
between them.

### 5.6 A confident failure, visible only through the explanation

Given *"As an AI language model, I don't have personal experiences"* — the
canonical AI tell — every model predicts **Human**, DistilBERT at 99.7%
confidence with all LIME weights below 0.002.

The explanation is distributional. HC3's AI class consists of long-form ChatGPT
answers averaging ~865 characters; a short first-person disclaimer resembles
nothing in the training distribution, so the model has no basis for a decision and
defaults confidently to the class that short informal text usually belongs to.

This is a genuine limitation of the pipeline, and LIME is what exposes it: the
near-zero attributions signal that the prediction rests on no identifiable
evidence. A high-confidence prediction with a flat explanation is a warning sign,
and it is visible only because the explanation was computed.

The demonstration set in `src/demo.py` therefore contains both short crafted
sentences and real HC3 excerpts. On in-distribution text the selected model's
peak weights reach 0.03–0.22; on the short sentences they fall to 0.006–0.013.
Running both groups makes the contrast visible rather than leaving it as a claim.

---

## 6. Limitations

1. **No deduplication.** Near-duplicate HC3 answers may straddle the train/test
   split, inflating all scores. The comparison stays fair; the absolute numbers
   are optimistic.
2. **Single-corpus, single-era evaluation.** HC3 is ChatGPT-era text from 2023.
   No Claude, Gemini or Llama output appears, and no model has been tested against
   a writer deliberately trying to evade detection.
3. **Out-of-distribution fragility.** As §5.6 shows, performance on short text
   bears no relation to the headline 0.9783.
4. **LIME is local and stochastic.** Explanations describe one prediction, not
   the model. The seed is fixed for reproducibility, but §5.3 shows the underlying
   variance is real and largest for the most confident model.
5. **`num_samples = 1000`, not LIME's default 5000.** This trades explanation
   fidelity for speed; at 10.6 s per explanation for DistilBERT the default would
   cost roughly 50 s each.
6. **The linguistic scale is coarse by design.** It cannot separate models within
   the same band, as the Logistic Regression / XGBoost tie demonstrates.
7. **Timings are single-machine.** All latency and explanation figures come from
   one Apple M4 using MPS. The ordering would hold elsewhere; the ratios would not.

---

## 7. Conclusion

Fine-tuning DistilBERT raised accuracy from **0.9117 to 0.9783**, with
non-overlapping confidence intervals and perfect recall on the AI class. It also
cost 61× the latency, 1,168× the storage and 208× the explanation time, and its
explanations proved the least stable of the four models.

TOPSIS and fuzzy TOPSIS both rank it last, and both rank XGBoost at or near the
top; XGBoost is selected. The recommendation survives every weighting tested and
holds across the entire practical range of the quality/cost trade-off,
surrendering only when cost is given literally no weight. The most accurate model
available is not the one this system should deploy — and stating that
conclusion with its reasoning exposed is the point of the exercise.

LIME then supplies what accuracy cannot: evidence about *why* a prediction was
made. Validated by a deletion test before being trusted, it showed the
transformer correctly reading register where bag-of-words models could only count
function words, and it exposed a confident failure on out-of-distribution input
that no test-set metric would have revealed.

---

## References

Chen, C.-T. (2000). Extensions of the TOPSIS for group decision-making under
fuzzy environment. *Fuzzy Sets and Systems*, 114(1), 1–9.

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
*Proceedings of KDD '16*, 785–794.

Guo, B., Zhang, X., Wang, Z., Jiang, M., Nie, J., Ding, Y., Yue, J., & Wu, Y.
(2023). How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and
Detection. *arXiv:2301.07597*.

Hwang, C.-L., & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and
Applications*. Springer-Verlag.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?":
Explaining the Predictions of Any Classifier. *Proceedings of KDD '16*,
1135–1144.

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled
version of BERT: smaller, faster, cheaper and lighter. *arXiv:1910.01108*.

Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.

---

## Reproducing

```bash
pip install -r requirements.txt        # plus: brew install libomp
python src/data_prep.py
python src/train.py && python src/train_candidates.py && python src/train_advanced.py
python src/benchmark.py && python src/select_model.py
python src/validate_lime.py && python src/make_figures.py
python src/demo.py --html && python src/compare.py
```

`python src/topsis.py` verifies the ranking arithmetic against two published
worked examples, independently of any model.
