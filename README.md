# Customer Churn Prediction

Binary classification on the Telco customer dataset (7,043 customers, 26.5% churn),
built as a leakage-free scikit-learn pipeline that goes from a raw spreadsheet row
to a scored churn probability.

The optimisation target is **recall on the churn class**: a missed churner costs a
whole customer lifetime, a false alarm costs one retention offer.

---

## Results

Held-out test set (1,409 customers, never touched until final evaluation):

| Metric | Value |
|---|---|
| ROC-AUC | **0.845** |
| PR-AUC | 0.654 |
| Recall (churn) | **0.802** |
| Precision (churn) | 0.507 |
| F1 (churn) | 0.621 |
| Balanced accuracy | 0.760 |

At the chosen operating point the model catches **300 of 374 churners (80.2%)**,
at the cost of 292 retention offers sent to customers who would have stayed.

Accuracy is 0.740 — *lower* than the 73.5% you would get by predicting "nobody
churns". That is the intended trade: the model converts useless majority-class
accuracy into churn recall.

### Model selection

Five-fold stratified CV on the training split, all three with imbalance correction:

| Model | CV ROC-AUC | CV Recall |
|---|---|---|
| **Logistic Regression** (tuned) | **0.847** | 0.799 |
| XGBoost | 0.841 | 0.755 |
| Random Forest | 0.838 | 0.618 |

Logistic regression wins, which is worth stating rather than hiding: this dataset
is close to linearly separable in log-odds, so the boosted model spends capacity
on interactions that are not there. Randomised search over 40 candidates moved
ROC-AUC by +0.0004 — the honest read is that the ceiling here is set by the
features, not the hyperparameters. The cheapest, most interpretable model winning
is a good outcome, not a disappointing one.

### What drives churn

Permutation importance over raw columns (mean drop in test ROC-AUC):

| Feature | Importance |
|---|---|
| tenure | 0.184 |
| Contract | 0.042 |
| InternetService | 0.035 |
| TotalCharges | 0.028 |

Direction, from the logistic coefficients as odds ratios:

- **Two-year contract → 0.21×** the odds of churn. The single strongest retention lever.
- **Fibre-optic internet → 2.45×**. High-price, high-expectation segment.
- **Electronic-check payment → 1.46×**. A low-commitment payment habit.
- **Every add-on service held** lowers risk; online security and tech support most.
- Churn is concentrated in the **first 12 months** — 43% for month-to-month
  customers versus 3% on two-year contracts.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

python churn_analysis.py        # full training run: figures + artifacts
python test_pipeline.py         # smoke tests
```

In VS Code, select `.venv` as the interpreter, open `churn_analysis.py` and run the
`# %%` cells in the Python Interactive Window.

### Scoring new customers

```python
import pandas as pd
from predict_churn import predict_churn

new_customers = pd.read_csv("new_customers.csv")   # raw schema, uncleaned
scored = predict_churn(new_customers)

scored[["churn_probability", "churn_prediction", "risk_band"]].head()
```

Or from the command line:

```bash
python predict_churn.py --input new_customers.csv --output scored.csv
```

`predict_churn` takes **raw** rows. Whitespace, blank `TotalCharges`, missing
values, feature engineering, scaling and encoding all happen inside the loaded
pipeline, so training and serving cannot drift apart. Categories never seen during
training are encoded as all-zero rather than raising, which keeps a batch job alive
when a new payment method shows up upstream.

---

## Layout

| File | Role |
|---|---|
| `churn_lib.py` | Config, loading, cleaning, feature engineering, pipeline factory |
| `viz.py` | Palette and chart functions |
| `evaluation.py` | Metrics, threshold selection, interpretability |
| `churn_analysis.py` | The `# %%` workflow: EDA → train → tune → evaluate → ship |
| `predict_churn.py` | Inference entry point and CLI |
| `test_pipeline.py` | Smoke tests for the pipeline contract |
| `artifacts/best_model.joblib` | Fitted end-to-end pipeline |
| `artifacts/model_metadata.json` | Model card: threshold, metrics, provenance |
| `reports/figures/*.png` | 17 charts |

`churn_lib.py` exists as a separate module for a specific reason: the saved
pipeline holds a reference to `engineer_features` by import path. Had that
function been defined in the analysis script, `joblib.load` would fail in any
process that did not re-run the whole notebook.

---

## Design decisions worth knowing

**Cleaning is split from learning.** `clean_frame` only does row-wise,
deterministic repairs — no statistic is learned from the data, so running it
before the train/test split cannot leak. Everything that *does* learn (medians,
modes, category levels, scaler means) lives inside the pipeline and is fitted on
training folds alone.

**`TotalCharges` is not median-imputed.** The 11 blank values all belong to
`tenure == 0` customers, so the correct value is `0.0` — they have not been billed
yet. Median imputation would invent about £1,400 of spend for the newest customers,
precisely the group the model most needs to read correctly. Median imputation
remains in the pipeline as the fallback for genuinely unknown values.

**`drop_duplicates` is off at inference.** De-duplication is right when building a
training set and wrong when scoring: a batch job must return exactly one row per
input row, in order, even when two customers share an identical feature vector.

**Tuning optimises ROC-AUC, not recall.** ROC-AUC is threshold-independent, so it
measures how well the model *ranks* customers by risk. Recall is then delivered by
choosing the decision threshold — the right lever, since moving the cut-off changes
recall without retraining. Optimising recall directly would have rewarded a model
that simply predicts "churn" for everyone.

**The threshold is chosen on out-of-fold training predictions**, never on the test
set. Picking it on test would quietly turn the held-out data into a validation set
and inflate every number above.

**On this dataset the tuned threshold lands at 0.499** — essentially the default.
That is because `class_weight='balanced'` has already re-weighted the classes, so
the 80%-recall point happens to sit at 0.5. The selection machinery still earns its
place: it is what lets the recall target be changed without retraining, and the
analysis prints the precision cost of moving the bar to 85%, 90% and 95%.

**A 10:1 cost model is shown but not adopted.** At FN=500 / FP=50 the
cost-minimising threshold is ~0.17, which would flag roughly half the base. It is
reported as a bound; the recall-floor rule keeps the campaign list small enough for
a retention team to actually work. Replace the cost constants with finance's real
numbers before quoting any of it externally.

---

## Limitations

- **This is a static snapshot, not a time series.** There are no dates, so the
  split is random rather than temporal, and the model cannot be validated against
  drift. A production version should be back-tested on a forward time window.
- **Precision is ~0.51 at the target recall.** Half the flagged customers would not
  have churned. That is acceptable when a retention offer is cheap and correct when
  a missed churner is expensive, but it makes the model unsuitable for anything
  costly or intrusive without raising the threshold.
- **`spend_trend_ratio` is undefined for `tenure == 0` customers** (no billing
  history to compare against) and is median-imputed for those 11 rows.
- **Fairness has not been audited.** `gender` and `SeniorCitizen` are in the
  feature set. Neither carries meaningful importance here, but a deployment that
  drives differential pricing or offers needs an explicit fairness review first.
