"""Standalone inference entry point for the churn model.

Importable::

    from predict_churn import predict_churn
    scored = predict_churn(new_customers_df)

Or from the command line::

    python predict_churn.py --input new_customers.csv --output scored.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from churn_lib import (
    METADATA_PATH,
    MODEL_PATH,
    REQUIRED_RAW_COLUMNS,
    TARGET,
    clean_frame,
)

#: Probability cut-offs for the qualitative band shown to the retention team.
RISK_BANDS: tuple[tuple[float, str], ...] = ((0.70, "High"), (0.40, "Medium"))
DEFAULT_THRESHOLD: float = 0.5


@lru_cache(maxsize=4)
def load_model(model_path: str | None = None) -> Pipeline:
    """Load the fitted pipeline from disk, cached across calls.

    Args:
        model_path: Override for the artifact location.

    Returns:
        The fitted end-to-end :class:`~sklearn.pipeline.Pipeline`.

    Raises:
        FileNotFoundError: If the artifact does not exist yet.
    """
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Run churn_analysis.py to train and persist one."
        )
    return joblib.load(path)


@lru_cache(maxsize=4)
def load_metadata(metadata_path: str | None = None) -> dict[str, Any]:
    """Load the model card, or return an empty dict when it is absent."""
    path = Path(metadata_path) if metadata_path else METADATA_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_threshold(override: float | None = None) -> float:
    """Return the decision threshold to apply.

    Precedence: explicit override, then the tuned value in the model card, then
    0.5. The tuned value is what the training run validated against the retention
    team's recall floor, so prefer it over the default in production.
    """
    if override is not None:
        return float(override)
    stored = load_metadata().get("decision_threshold", {})
    return float(stored.get("threshold", DEFAULT_THRESHOLD))


def _assign_band(probability: float) -> str:
    """Map a probability to a High / Medium / Low retention-priority band."""
    for cutoff, label in RISK_BANDS:
        if probability >= cutoff:
            return label
    return "Low"


def validate_input(df: pd.DataFrame) -> None:
    """Check that the raw columns the pipeline needs are present.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: If the frame is empty or required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("Input frame is empty")

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def predict_churn(
    new_data_df: pd.DataFrame,
    *,
    threshold: float | None = None,
    model_path: str | None = None,
    include_input: bool = True,
) -> pd.DataFrame:
    """Score raw customer records for churn risk.

    Accepts the same raw schema as the training data -- the saved pipeline handles
    whitespace, missing values, feature engineering, scaling and encoding, so no
    preprocessing is required or wanted from the caller. Categories never seen in
    training are encoded as all-zero rather than raising, which keeps a batch job
    alive when a new payment method appears upstream.

    Args:
        new_data_df: Raw customer rows. Must contain
            :data:`~churn_lib.REQUIRED_RAW_COLUMNS`; extra columns are ignored and
            passed through to the output.
        threshold: Probability cut-off for the positive class. Defaults to the
            tuned value stored in the model card.
        model_path: Override for the model artifact location.
        include_input: Return the input columns alongside the predictions.

    Returns:
        A frame indexed like the input with:
            ``churn_probability`` -- predicted probability of churn (float).
            ``churn_prediction``  -- 0/1 at the applied threshold.
            ``churn_label``       -- ``"Yes"`` / ``"No"``.
            ``risk_band``         -- ``"High"`` / ``"Medium"`` / ``"Low"``.
            ``threshold``         -- the cut-off actually applied.

    Raises:
        ValueError: If required columns are missing or the frame is empty.
        FileNotFoundError: If the model artifact has not been trained yet.

    Example:
        >>> scored = predict_churn(new_customers)  # doctest: +SKIP
        >>> scored.loc[scored["risk_band"] == "High", "churn_probability"].mean()  # doctest: +SKIP
    """
    validate_input(new_data_df)

    pipeline = load_model(model_path)
    applied_threshold = resolve_threshold(threshold)

    # Drop any label column so a leaked target cannot reach the pipeline, then
    # apply the same deterministic cleaning used at training time. De-duplication
    # is switched off here: scoring must return one row per input row, in order,
    # even when two customers happen to share an identical feature vector.
    features = new_data_df.drop(columns=[TARGET], errors="ignore")
    cleaned = clean_frame(features, drop_duplicates=False)

    probabilities = pipeline.predict_proba(cleaned)[:, 1]
    predictions = (probabilities >= applied_threshold).astype(int)

    result = pd.DataFrame(
        {
            "churn_probability": probabilities.round(4),
            "churn_prediction": predictions,
            "churn_label": pd.Series(predictions, index=new_data_df.index).map({0: "No", 1: "Yes"}),
            "risk_band": [_assign_band(p) for p in probabilities],
            "threshold": applied_threshold,
        },
        index=new_data_df.index,
    )

    if include_input:
        return pd.concat([new_data_df, result], axis=1)
    return result


def _read_any(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper: score a file of raw customers and write the results out."""
    parser = argparse.ArgumentParser(description="Score customers for churn risk.")
    parser.add_argument("--input", required=True, type=Path, help="CSV/Excel of raw customers")
    parser.add_argument("--output", type=Path, help="Where to write scored CSV (default: stdout)")
    parser.add_argument("--threshold", type=float, help="Override the stored decision threshold")
    parser.add_argument("--model", type=str, help="Override the model artifact path")
    args = parser.parse_args(argv)

    frame = _read_any(args.input)
    scored = predict_churn(frame, threshold=args.threshold, model_path=args.model)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.output, index=False)
        high = int((scored["risk_band"] == "High").sum())
        print(
            f"Scored {len(scored):,} customers at threshold {scored['threshold'].iloc[0]:.3f} "
            f"({int(scored['churn_prediction'].sum()):,} flagged, {high:,} high-risk) "
            f"-> {args.output}"
        )
    else:
        cols = ["churn_probability", "churn_prediction", "churn_label", "risk_band"]
        print(scored[cols].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
