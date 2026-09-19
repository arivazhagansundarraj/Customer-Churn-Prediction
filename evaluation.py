"""Evaluation, threshold selection and interpretability helpers.

Churn is an asymmetric problem: a missed churner costs a whole customer
lifetime, while a false alarm costs one retention offer. These helpers keep that
asymmetry explicit rather than hiding it behind a default 0.5 cut-off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from churn_lib import RANDOM_STATE, get_encoded_feature_names


@dataclass(frozen=True)
class ThresholdChoice:
    """A selected operating point and the trade-off it implies."""

    threshold: float
    precision: float
    recall: float
    f1: float
    rule: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the operating point."""
        return {
            "threshold": round(float(self.threshold), 4),
            "precision": round(float(self.precision), 4),
            "recall": round(float(self.recall), 4),
            "f1": round(float(self.f1), 4),
            "rule": self.rule,
        }


def score_predictions(
    y_true: Sequence[int], y_pred: Sequence[int], y_proba: Sequence[float]
) -> dict[str, float]:
    """Compute the headline metrics for one set of predictions.

    ``roc_auc`` and ``pr_auc`` are threshold-independent; the rest describe the
    specific operating point that produced ``y_pred``.
    """
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def sweep_thresholds(
    y_true: Sequence[int], y_proba: Sequence[float], *, steps: int = 199
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Evaluate precision, recall and F1 across a grid of decision thresholds.

    Returns:
        ``(thresholds, {"Precision": ..., "Recall": ..., "F1": ...})``.
    """
    thresholds = np.linspace(0.01, 0.99, steps)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    precision, recall, f1 = [], [], []
    for cut in thresholds:
        y_pred = (y_proba >= cut).astype(int)
        precision.append(precision_score(y_true, y_pred, zero_division=0))
        recall.append(recall_score(y_true, y_pred, zero_division=0))
        f1.append(f1_score(y_true, y_pred, zero_division=0))

    return thresholds, {
        "Precision": np.asarray(precision),
        "Recall": np.asarray(recall),
        "F1": np.asarray(f1),
    }


def choose_threshold(
    y_true: Sequence[int], y_proba: Sequence[float], *, min_recall: float = 0.80
) -> ThresholdChoice:
    """Pick the decision threshold that maximises precision at a recall floor.

    The business rule comes first: the retention team must catch at least
    ``min_recall`` of churners. Among all thresholds that clear that bar, the one
    with the best precision wastes the fewest retention offers. If the floor is
    unreachable, this falls back to the best-F1 threshold and says so in ``rule``.

    Args:
        y_true: Ground-truth labels.
        y_proba: Predicted churn probabilities.
        min_recall: Minimum acceptable recall on the churn class.

    Returns:
        The chosen :class:`ThresholdChoice`.
    """
    precision, recall, cuts = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one more point than thresholds; drop it.
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    eligible = np.flatnonzero(recall >= min_recall)
    if eligible.size:
        best = eligible[int(np.argmax(precision[eligible]))]
        rule = f"max precision subject to recall >= {min_recall:.0%}"
    else:
        best = int(np.argmax(f1))
        rule = f"max F1 (recall floor of {min_recall:.0%} unreachable)"

    return ThresholdChoice(
        threshold=float(cuts[best]),
        precision=float(precision[best]),
        recall=float(recall[best]),
        f1=float(f1[best]),
        rule=rule,
    )


def expected_cost(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    *,
    cost_false_negative: float = 500.0,
    cost_false_positive: float = 50.0,
    steps: int = 199,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cost curve over thresholds under an explicit FN/FP cost assumption.

    Defaults encode "a lost customer is worth ten retention offers". Replace them
    with the finance team's real numbers before quoting any figure externally.

    Returns:
        ``(thresholds, cost_per_customer, cost_minimising_threshold)``.
    """
    thresholds = np.linspace(0.01, 0.99, steps)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)

    costs = []
    for cut in thresholds:
        y_pred = (y_proba >= cut).astype(int)
        false_neg = int(np.sum((y_true == 1) & (y_pred == 0)))
        false_pos = int(np.sum((y_true == 0) & (y_pred == 1)))
        costs.append((false_neg * cost_false_negative + false_pos * cost_false_positive) / n)

    costs = np.asarray(costs)
    return thresholds, costs, float(thresholds[int(np.argmin(costs))])


def report_classification(
    y_true: Sequence[int], y_pred: Sequence[int], *, digits: int = 3
) -> str:
    """Return a labelled ``classification_report`` for the two churn classes."""
    return classification_report(
        y_true, y_pred, target_names=["Retained (0)", "Churned (1)"],
        digits=digits, zero_division=0,
    )


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    """Return the 2x2 confusion matrix with rows ordered ``[retained, churned]``."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def roc_points(
    y_true: Sequence[int], y_proba: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(fpr, tpr, roc_auc)`` ready for plotting."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return fpr, tpr, float(roc_auc_score(y_true, y_proba))


def pr_points(
    y_true: Sequence[int], y_proba: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(recall, precision, average_precision)`` ready for plotting."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return recall, precision, float(average_precision_score(y_true, y_proba))


# --------------------------------------------------------------------------- #
# Interpretability
# --------------------------------------------------------------------------- #
def model_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Pull the classifier's own importances or coefficients, with encoded names.

    Args:
        pipeline: A fitted project pipeline.

    Returns:
        Frame of ``feature`` / ``importance`` / ``signed`` sorted by magnitude.
        ``signed`` is ``True`` for linear coefficients, where direction is
        meaningful, and ``False`` for tree-based gain importances.

    Raises:
        AttributeError: If the classifier exposes neither attribute.
    """
    classifier = pipeline.named_steps["classifier"]
    names = get_encoded_feature_names(pipeline)

    if hasattr(classifier, "coef_"):
        values = np.ravel(classifier.coef_)
        signed = True
    elif hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
        signed = False
    else:
        raise AttributeError(
            f"{type(classifier).__name__} exposes no coef_ or feature_importances_"
        )

    return (
        pd.DataFrame({"feature": names, "importance": values})
        .assign(signed=signed)
        .reindex(columns=["feature", "importance", "signed"])
        .sort_values("importance", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )


def raw_permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: Sequence[int],
    *,
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Permutation importance over the *raw* input columns.

    Permuting before the pipeline runs means each score is attributable to a
    business-legible column ("Contract", "tenure") rather than to a one-hot
    fragment, and it captures the derived features a column feeds into.

    Args:
        pipeline: Fitted project pipeline.
        X: Raw feature frame (typically the held-out test set).
        y: True labels for ``X``.
        scoring: Metric to degrade; ROC-AUC is threshold-independent.
        n_repeats: Shuffles per column.
        random_state: Seed for reproducibility.
        n_jobs: Parallel workers; ``None`` keeps it single-process.

    Returns:
        Frame of ``feature`` / ``importance_mean`` / ``importance_std``, best first.
    """
    result = permutation_importance(
        pipeline, X, y, scoring=scoring, n_repeats=n_repeats,
        random_state=random_state, n_jobs=n_jobs,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
