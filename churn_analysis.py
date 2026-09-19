"""End-to-end customer-churn workflow: EDA -> model -> evaluate -> ship.

Run it top to bottom as a script, or step through the ``# %%`` cells in the
VS Code Python Interactive Window.

Outputs:
    reports/figures/*.png        charts for the stakeholder deck
    artifacts/best_model.joblib  the fitted end-to-end pipeline
    artifacts/model_metadata.json  chosen threshold, metrics, provenance
"""

# %% [markdown]
# # Customer Churn Prediction
#
# **Objective.** Flag customers likely to churn early enough to intervene.
# Recall on the churn class is the metric that matters: a missed churner costs a
# whole customer lifetime, a false alarm costs one retention offer.
#
# **Method.** Every transformation lives inside a scikit-learn `Pipeline`, so the
# test set and future production traffic are scored by statistics learned on
# training data alone.

# %%
# --- Setup -------------------------------------------------------------------
from __future__ import annotations

import json
import platform
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

import churn_lib as cl
import evaluation as ev
import viz

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_XGBOOST = False
    warnings.warn("xgboost not installed; benchmarking Logistic Regression + Random Forest only")

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
viz.apply_style()
cl.ensure_directories()

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=cl.RANDOM_STATE)
MIN_RECALL = 0.80  # retention team's coverage requirement

print(f"python {platform.python_version()} | pandas {pd.__version__} | xgboost: {HAS_XGBOOST}")


# %% [markdown]
# ## 1. Data loading & quality audit

# %%
raw = cl.load_raw_data()
print(f"Loaded {cl.resolve_data_path()}")
print(f"Shape: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
raw.head()

# %%
# Structure and dtypes. TotalCharges is the usual offender: it ships as text
# because ~11 brand-new customers carry a blank string instead of a number.
audit = pd.DataFrame(
    {
        "dtype": raw.dtypes.astype(str),
        "n_unique": raw.nunique(),
        "n_missing": raw.isna().sum(),
        "pct_missing": (raw.isna().mean() * 100).round(2),
    }
)
audit["n_blank_strings"] = [
    int(raw[c].astype("string").str.strip().eq("").sum()) if raw[c].dtype == object else 0
    for c in raw.columns
]
audit

# %%
# Numeric summary before cleaning.
raw.describe().T.round(2)

# %% [markdown]
# ## 2. Cleaning
#
# `clean_frame` is deliberately limited to row-wise, deterministic fixes -- no
# statistic is learned from the data, so applying it before the split cannot leak
# test information. Anything that *does* learn (medians, modes, scaling, category
# levels) is deferred to the pipeline.

# %%
df = cl.clean_frame(raw)
df["churn_flag"] = cl.encode_target(df[cl.TARGET])

remaining_nulls = df.isna().sum().loc[lambda s: s > 0]
print(f"Rows: {len(raw):,} -> {len(df):,} after de-duplication")
print("Remaining missing values: none" if remaining_nulls.empty
      else f"Remaining missing values:\n{remaining_nulls.to_string()}")

# The 11 blank TotalCharges all belong to tenure-0 customers, so 0.0 is the
# correct value -- median imputation would have invented ~1.4k of spend for them.
brand_new = df.loc[df["tenure"].eq(0), ["tenure", "MonthlyCharges", "TotalCharges", cl.TARGET]]
print(f"\nTenure-0 customers repaired: {len(brand_new)}")
brand_new.head()

# %% [markdown]
# ## 3. Exploratory analysis

# %%
# Class balance -- the headline constraint on how this model must be evaluated.
churn_rate = df["churn_flag"].mean()
counts = df["churn_flag"].value_counts().sort_index()
print(f"Retained: {counts[0]:,}   Churned: {counts[1]:,}   Churn rate: {churn_rate:.2%}")
print(f"Imbalance ratio: {counts[0] / counts[1]:.2f} : 1")

fig = viz.plot_target_balance(df["churn_flag"])
viz.save(fig, "01_class_balance")

# %%
# Churn rate by contract type -- the single strongest lever in this dataset.
fig = viz.plot_churn_rate_by_category(
    df, "Contract", title="Churn rate by contract type"
)
viz.save(fig, "02_churn_by_contract")

# %%
# Tenure: churners are overwhelmingly concentrated in the first year.
fig = viz.plot_distribution_by_churn(
    df, "tenure", title="Tenure distribution by outcome", xlabel="Tenure (months)"
)
viz.save(fig, "03_tenure_distribution")

fig = viz.plot_churn_curve(
    cl.engineer_features(df), "tenure_bucket",
    title="Churn rate decays sharply with tenure", xlabel="Tenure bucket",
)
viz.save(fig, "04_churn_by_tenure_bucket")

# %%
# Monthly charges: risk climbs with the bill, concentrated in the fibre segment.
fig = viz.plot_distribution_by_churn(
    df, "MonthlyCharges", title="Monthly charges by outcome", xlabel="Monthly charges"
)
viz.save(fig, "05_monthly_charges_distribution")

# %%
# Remaining service and billing drivers.
for name, column in [
    ("06_churn_by_internet", "InternetService"),
    ("07_churn_by_payment", "PaymentMethod"),
    ("08_churn_by_tech_support", "TechSupport"),
]:
    fig = viz.plot_churn_rate_by_category(df, column)
    viz.save(fig, name)

# %%
# Segment table for the appendix: churn rate and volume per level, worst first.
segments = []
for column in cl.CATEGORICAL_FEATURES:
    if column not in df.columns:
        continue
    grouped = df.groupby(column, observed=True)["churn_flag"].agg(["mean", "size"])
    for level, (rate, size) in grouped.iterrows():
        segments.append(
            {"feature": column, "level": level, "churn_rate": rate, "customers": int(size)}
        )

segment_table = (
    pd.DataFrame(segments)
    .assign(lift=lambda d: d["churn_rate"] / churn_rate)
    .sort_values("churn_rate", ascending=False)
    .reset_index(drop=True)
)
segment_table.head(12).round(3)

# %% [markdown]
# ## 4. Train / test split
#
# Stratified so both halves carry the same 26.5% churn rate, and fixed at
# `random_state=42` so the numbers below reproduce exactly.

# %%
model_frame = df.drop(columns=["churn_flag"])
X_train, X_test, y_train, y_test = cl.split_data(model_frame)

print(f"Train: {X_train.shape[0]:,} rows  ({y_train.mean():.2%} churn)")
print(f"Test:  {X_test.shape[0]:,} rows  ({y_test.mean():.2%} churn)")
print(f"Features in: {X_train.shape[1]} raw columns")

# Imbalance correction for the gradient-boosted model.
scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
print(f"scale_pos_weight: {scale_pos_weight:.3f}")

# %% [markdown]
# ## 5. Baseline model comparison
#
# All three carry an imbalance correction (`class_weight='balanced'` for the
# scikit-learn models, `scale_pos_weight` for XGBoost), so recall is not sacrificed
# to the majority class before tuning even starts.

# %%
candidates: dict[str, object] = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=cl.RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=cl.RANDOM_STATE,
        n_jobs=-1,
    ),
}
if HAS_XGBOOST:
    candidates["XGBoost"] = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=cl.RANDOM_STATE,
        n_jobs=-1,
    )

SCORING = ["roc_auc", "average_precision", "recall", "precision", "f1", "balanced_accuracy"]

rows = []
for name, estimator in candidates.items():
    pipeline = cl.build_pipeline(estimator)
    cv_results = cross_validate(pipeline, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=1)
    rows.append({"model": name, **{m: cv_results[f"test_{m}"].mean() for m in SCORING}})
    print(f"{name:<22} roc_auc={rows[-1]['roc_auc']:.4f}  recall={rows[-1]['recall']:.4f}")

benchmark = pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)
benchmark.round(4)

# %%
fig = viz.plot_model_comparison(benchmark, metric="roc_auc")
viz.save(fig, "09_model_comparison_roc_auc")

fig = viz.plot_model_comparison(benchmark, metric="recall")
viz.save(fig, "10_model_comparison_recall")

# %% [markdown]
# ## 6. Hyperparameter tuning
#
# The search refits on **ROC-AUC**: it is threshold-independent, so it measures how
# well the model *ranks* customers by risk. Recall is tracked alongside, then
# delivered in section 7 by choosing the decision threshold -- which is the right
# lever for it, since moving the cut-off changes recall without retraining.

# %%
best_name = benchmark.index[0]
print(f"Tuning: {best_name}")

if best_name == "XGBoost":
    search_space = {
        "classifier__n_estimators": [200, 300, 400, 600, 800],
        "classifier__learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1],
        "classifier__max_depth": [2, 3, 4, 5, 6],
        "classifier__min_child_weight": [1, 3, 5, 8],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__gamma": [0, 0.1, 0.3, 0.5],
        "classifier__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }
elif best_name == "Random Forest":
    search_space = {
        "classifier__n_estimators": [300, 500, 800],
        "classifier__max_depth": [None, 6, 10, 16, 24],
        "classifier__min_samples_leaf": [1, 2, 4, 8, 16],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__max_features": ["sqrt", "log2", 0.4, 0.6],
    }
else:
    # scikit-learn >= 1.8 expresses the penalty through l1_ratio: 0.0 is ridge,
    # 1.0 is lasso, anything between is elastic net and needs the saga solver.
    search_space = [
        {
            "classifier__C": np.logspace(-3, 2, 20),
            "classifier__l1_ratio": [0.0, 1.0],
            "classifier__solver": ["liblinear"],
        },
        {
            "classifier__C": np.logspace(-3, 2, 20),
            "classifier__l1_ratio": [0.15, 0.35, 0.5, 0.65, 0.85],
            "classifier__solver": ["saga"],
        },
    ]

search = RandomizedSearchCV(
    estimator=cl.build_pipeline(candidates[best_name]),
    param_distributions=search_space,
    n_iter=40,
    scoring={"roc_auc": "roc_auc", "recall": "recall", "average_precision": "average_precision"},
    refit="roc_auc",
    cv=CV,
    random_state=cl.RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
    return_train_score=False,
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
best_index = search.best_index_
print(f"\nBest CV ROC-AUC: {search.best_score_:.4f}")
print(f"Recall at that setting: {search.cv_results_['mean_test_recall'][best_index]:.4f}")
print("\nBest parameters:")
for key, value in sorted(search.best_params_.items()):
    print(f"  {key.replace('classifier__', ''):<22} {value}")

# %%
# Did tuning actually help? Compare against the untuned baseline.
tuning_gain = search.best_score_ - benchmark.loc[best_name, "roc_auc"]
print(f"ROC-AUC: {benchmark.loc[best_name, 'roc_auc']:.4f} -> {search.best_score_:.4f} "
      f"({tuning_gain:+.4f})")

# %% [markdown]
# ## 7. Choosing the operating point
#
# The threshold is selected on **out-of-fold training predictions**, never on the
# test set -- picking it on test would quietly turn the held-out data into a
# validation set and inflate every number in section 8.

# %%
oof_proba = cross_val_predict(
    best_model, X_train, y_train, cv=CV, method="predict_proba", n_jobs=1
)[:, 1]

thresholds, sweep = ev.sweep_thresholds(y_train, oof_proba)
choice = ev.choose_threshold(y_train, oof_proba, min_recall=MIN_RECALL)

print(f"Rule:      {choice.rule}")
print(f"Threshold: {choice.threshold:.3f}")
print(f"Out-of-fold  recall={choice.recall:.3f}  "
      f"precision={choice.precision:.3f}  f1={choice.f1:.3f}")
print(f"(default 0.5 would give recall={sweep['Recall'][np.argmin(abs(thresholds - 0.5))]:.3f})")

fig = viz.plot_threshold_sweep(thresholds, sweep, chosen=choice.threshold)
viz.save(fig, "11_threshold_sweep")

# %%
# Worth stating plainly: because `class_weight='balanced'` already re-weights the
# classes, the 80%-recall point lands almost exactly on the default 0.5, so the
# tuned threshold barely moves. The machinery is not redundant -- it is what lets
# the recall target be changed without retraining. The cost of raising the bar:
for target in (0.75, 0.80, 0.85, 0.90, 0.95):
    alt = ev.choose_threshold(y_train, oof_proba, min_recall=target)
    flagged = float(np.mean(oof_proba >= alt.threshold))
    print(f"  recall >= {target:.0%} -> threshold {alt.threshold:.3f}   "
          f"precision {alt.precision:.3f}   flags {flagged:.1%} of the base")

# %%
# Cross-check against an explicit cost model. Swap in finance's real numbers
# before quoting any of this externally.
cost_thresholds, costs, cost_optimal = ev.expected_cost(
    y_train, oof_proba, cost_false_negative=500.0, cost_false_positive=50.0
)


def cost_at(cut: float) -> float:
    """Expected cost per customer at the threshold nearest ``cut``."""
    return float(costs[int(np.argmin(np.abs(cost_thresholds - cut)))])


print("Expected cost per customer (FN=500, FP=50):")
print(f"  default 0.500            -> {cost_at(0.5):6.2f}")
print(f"  chosen  {choice.threshold:.3f}            -> {cost_at(choice.threshold):6.2f}")
print(f"  cost-optimal {cost_optimal:.3f}       -> {cost_at(cost_optimal):6.2f}")
print(
    "\nThe cost-optimal cut sits far lower because a 10:1 penalty says to flag almost\n"
    "anyone plausible. It is shown as a bound, not adopted: the recall-floor rule keeps\n"
    "the campaign list small enough for the retention team to actually work."
)

# %% [markdown]
# ## 8. Test-set evaluation
#
# First and only look at the held-out 20%.

# %%
test_proba = best_model.predict_proba(X_test)[:, 1]
test_pred_default = (test_proba >= 0.5).astype(int)
test_pred_tuned = (test_proba >= choice.threshold).astype(int)

metrics_default = ev.score_predictions(y_test, test_pred_default, test_proba)
metrics_tuned = ev.score_predictions(y_test, test_pred_tuned, test_proba)

comparison = pd.DataFrame(
    {"threshold 0.500": metrics_default, f"threshold {choice.threshold:.3f}": metrics_tuned}
)
print(comparison.round(4).to_string())

# %%
print(f"=== Classification report @ threshold {choice.threshold:.3f} ===\n")
print(ev.report_classification(y_test, test_pred_tuned))

cm = ev.confusion(y_test, test_pred_tuned)
tn, fp, fn, tp = cm.ravel()
print(f"True negatives : {tn:>5,}   False positives: {fp:>5,}")
print(f"False negatives: {fn:>5,}   True positives : {tp:>5,}")
print(f"\nChurners caught: {tp:,} of {tp + fn:,} ({tp / (tp + fn):.1%})")
print(f"Offers wasted on non-churners: {fp:,}")

fig = viz.plot_confusion_matrix(
    cm, title=f"Confusion matrix -- {best_name} @ {choice.threshold:.3f}"
)
viz.save(fig, "12_confusion_matrix")

# %%
# ROC and precision-recall curves for every candidate, refit on the full train set.
roc_curves, pr_curves = {}, {}
for name, estimator in candidates.items():
    pipeline = (
        best_model if name == best_name
        else cl.build_pipeline(estimator).fit(X_train, y_train)
    )
    proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, auc = ev.roc_points(y_test, proba)
    rec, prec, ap = ev.pr_points(y_test, proba)
    label = f"{name} (tuned)" if name == best_name else name
    roc_curves[label] = (fpr, tpr, auc)
    pr_curves[label] = (rec, prec, ap)

fig = viz.plot_roc_pr(roc_curves, kind="roc")
viz.save(fig, "13_roc_curves")

fig = viz.plot_roc_pr(pr_curves, kind="pr", positive_rate=float(y_test.mean()))
viz.save(fig, "14_precision_recall_curves")

# %% [markdown]
# ## 9. What drives churn
#
# Two views, because they answer different questions. Permutation importance over
# the raw columns tells the business *which fields matter*; the logistic
# coefficients tell it *in which direction*.

# %%
perm = ev.raw_permutation_importance(best_model, X_test, y_test, scoring="roc_auc", n_repeats=10)
print(perm.head(12).round(4).to_string(index=False))

fig = viz.plot_feature_importance(
    perm["feature"], perm["importance_mean"], top_n=15,
    title=f"Permutation importance -- {best_name} (drop in test ROC-AUC)",
    xlabel="Mean decrease in ROC-AUC",
)
viz.save(fig, "15_permutation_importance")

# %%
# The model's own importances, at one-hot granularity.
internal = ev.model_feature_importance(best_model)
print(internal.head(12).round(4).to_string(index=False))

fig = viz.plot_feature_importance(
    internal["feature"], internal["importance"], top_n=15,
    title=f"{best_name} internal feature importance",
    xlabel="Coefficient" if bool(internal["signed"].iloc[0]) else "Gain importance",
    signed=bool(internal["signed"].iloc[0]),
)
viz.save(fig, "16_model_feature_importance")

# %%
# Signed direction for the stakeholder deck. Tree importances say how *much* a
# feature matters but not which way it pushes, so when the winner is a tree model
# we fit a logistic surrogate on the same pipeline purely to read off direction.
# When the winner is already linear, reuse it rather than fitting a second copy.
if bool(internal["signed"].iloc[0]):
    interpretable = best_model
    print(f"{best_name} is linear -- reading coefficients directly.")
else:
    interpretable = cl.build_pipeline(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cl.RANDOM_STATE)
    ).fit(X_train, y_train)
    surrogate_auc = ev.roc_points(y_test, interpretable.predict_proba(X_test)[:, 1])[2]
    print(f"Logistic surrogate test ROC-AUC {surrogate_auc:.4f} "
          f"vs {best_name} {metrics_tuned['roc_auc']:.4f}")

coefficients = ev.model_feature_importance(interpretable).assign(
    odds_ratio=lambda d: np.exp(d["importance"])
)
print("Top churn drivers (odds ratio > 1 raises risk):")
summary_cols = ["feature", "importance", "odds_ratio"]
print(coefficients.head(12)[summary_cols].round(3).to_string(index=False))

fig = viz.plot_feature_importance(
    coefficients["feature"], coefficients["importance"], top_n=15,
    title="Direction of effect -- logistic regression coefficients",
    xlabel="Log-odds coefficient", signed=True,
)
viz.save(fig, "17_logistic_coefficients")

# %% [markdown]
# ## 10. Persist the pipeline
#
# `best_model.joblib` holds the whole raw-in / prediction-out pipeline: feature
# engineering, imputation, scaling, encoding and the classifier. The metadata
# sidecar carries the chosen threshold and the test metrics, so a serving process
# never has to guess the operating point.

# %%
joblib.dump(best_model, cl.MODEL_PATH)

metadata = {
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "model": best_name,
    "best_params": {k.replace("classifier__", ""): str(v) for k, v in search.best_params_.items()},
    "decision_threshold": choice.as_dict(),
    "cv_roc_auc": round(float(search.best_score_), 4),
    "test_metrics_at_threshold": {k: round(float(v), 4) for k, v in metrics_tuned.items()},
    "test_metrics_at_050": {k: round(float(v), 4) for k, v in metrics_default.items()},
    "confusion_matrix_at_threshold": {
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    },
    "training_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "train_churn_rate": round(float(y_train.mean()), 4),
    "required_raw_columns": cl.REQUIRED_RAW_COLUMNS,
    "top_drivers": perm.head(10)["feature"].tolist(),
    "random_state": cl.RANDOM_STATE,
}
cl.METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print(f"Model    -> {cl.MODEL_PATH}")
print(f"Metadata -> {cl.METADATA_PATH}")

# %% [markdown]
# ## 11. Inference smoke test
#
# Reload from disk and score raw rows through the public entry point, exactly as
# a batch job or API would.

# %%
from predict_churn import predict_churn  # noqa: E402  (imported after the model exists)

sample = X_test.head(5).copy()
scored = predict_churn(sample)
print(scored[["churn_probability", "churn_prediction", "risk_band", "threshold"]].to_string())

print(f"\nActual outcomes: {y_test.head(5).tolist()}")

# %%
# Batch scoring the whole test set, ranked by risk -- the shape of the weekly
# hand-off to the retention team.
ranked = predict_churn(X_test).sort_values("churn_probability", ascending=False)
print(f"High-risk customers: {(ranked['risk_band'] == 'High').sum():,} of {len(ranked):,}")
print(ranked["risk_band"].value_counts().to_string())
ranked.head(10)[["tenure", "Contract", "MonthlyCharges", "churn_probability", "risk_band"]]

# %% [markdown]
# ## Summary
#
# Run the cells above and read the printed numbers; the figures land in
# `reports/figures/`. The model card in `artifacts/model_metadata.json` records the
# operating point, so retraining is a matter of re-running this file, not
# re-deriving the decisions behind it.
