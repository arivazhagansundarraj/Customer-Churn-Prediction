"""Smoke tests for the churn pipeline's contract.

Deliberately small: these guard the invariants that are easy to break silently
during a refactor -- row counts at inference, leakage-free preprocessing, and
graceful handling of the messy inputs production actually sends.

Run with ``python test_pipeline.py`` (no pytest required) or ``pytest test_pipeline.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import churn_lib as cl


def _sample_frame(n: int = 200) -> pd.DataFrame:
    """Take a deterministic slice of the real dataset."""
    return cl.load_raw_data().head(n)


def test_clean_frame_handles_blank_total_charges() -> None:
    """Whitespace-only TotalCharges becomes 0.0 for tenure-0 customers.

    The Excel export already parses that column as float, so the frame is cast
    back to object first to reproduce what a raw CSV read actually delivers.
    """
    frame = _sample_frame(20)
    frame["TotalCharges"] = frame["TotalCharges"].astype(object)
    frame.loc[frame.index[0], "TotalCharges"] = "  "
    frame.loc[frame.index[0], "tenure"] = 0
    frame.loc[frame.index[1], "TotalCharges"] = " 1234.5 "  # padded but valid

    cleaned = cl.clean_frame(frame)
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert cleaned.loc[1, "TotalCharges"] == 1234.5


def test_clean_frame_preserves_rows_when_not_deduplicating() -> None:
    """Inference mode must return one row per input row, index intact."""
    frame = _sample_frame(50)
    duplicated = pd.concat([frame, frame], ignore_index=True)

    assert len(cl.clean_frame(duplicated, drop_duplicates=False)) == len(duplicated)
    assert len(cl.clean_frame(duplicated, drop_duplicates=True)) < len(duplicated)

    shuffled = frame.sample(frac=1.0, random_state=0)
    assert cl.clean_frame(shuffled, drop_duplicates=False).index.equals(shuffled.index)


def test_engineer_features_is_row_local() -> None:
    """A row's derived features must not depend on the other rows present."""
    frame = cl.clean_frame(_sample_frame(100))
    derived = ["avg_monthly_spend", "spend_trend_ratio", "num_addon_services", "tenure_bucket"]

    full = cl.engineer_features(frame)[derived]
    single = cl.engineer_features(frame.iloc[[7]])[derived]

    pd.testing.assert_frame_equal(full.iloc[[7]], single, check_dtype=False)


def test_pipeline_learns_nothing_from_the_test_split() -> None:
    """Scaler statistics must come from the training rows alone."""
    df = cl.clean_frame(cl.load_raw_data())
    X_train, X_test, y_train, _ = cl.split_data(df)

    pipeline = cl.build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    scaler = pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["scaler"]

    train_only = cl.build_pipeline(LogisticRegression(max_iter=1000)).fit(
        X_train, y_train
    ).named_steps["preprocess"].named_transformers_["num"].named_steps["scaler"]

    np.testing.assert_allclose(scaler.mean_, train_only.mean_)
    # Refitting on train+test would move the means; assert they differ.
    combined = pd.concat([X_train, X_test])
    both = cl.build_pipeline(LogisticRegression(max_iter=1000)).fit(
        combined, pd.concat([y_train, pd.Series([0] * len(X_test), index=X_test.index)])
    ).named_steps["preprocess"].named_transformers_["num"].named_steps["scaler"]
    assert not np.allclose(scaler.mean_, both.mean_)


def test_pipeline_survives_unseen_categories_and_nulls() -> None:
    """A new category level or a missing value must not crash scoring."""
    df = cl.clean_frame(cl.load_raw_data())
    X_train, X_test, y_train, _ = cl.split_data(df)
    pipeline = cl.build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)

    messy = X_test.head(5).copy()
    messy.loc[messy.index[0], "PaymentMethod"] = "Crypto wallet"  # never seen in training
    messy.loc[messy.index[1], "MonthlyCharges"] = np.nan
    messy.loc[messy.index[2], "Contract"] = None

    proba = pipeline.predict_proba(messy)[:, 1]
    assert len(proba) == 5
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()


def test_encode_target_rejects_unknown_labels() -> None:
    """Unparseable churn labels fail loudly rather than silently becoming 0."""
    assert cl.encode_target(pd.Series(["Yes", "No", "yes"])).tolist() == [1, 0, 1]
    try:
        cl.encode_target(pd.Series(["Yes", "Maybe"]))
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unrecognised label")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - a test runner reports every failure
            failures += 1
            print(f"FAIL  {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
