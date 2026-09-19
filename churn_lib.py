"""Core data and modelling utilities for the Telco customer-churn project.

Everything that must survive a ``joblib`` round-trip lives here. The fitted
pipeline references :func:`engineer_features` by import path, so this module has
to be importable from any process that loads ``artifacts/best_model.joblib``.

Typical use::

    from churn_lib import load_raw_data, clean_frame, split_data, build_pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Project layout & constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "Dataset"
ARTIFACT_DIR: Final[Path] = PROJECT_ROOT / "artifacts"
FIGURE_DIR: Final[Path] = PROJECT_ROOT / "reports" / "figures"
MODEL_PATH: Final[Path] = ARTIFACT_DIR / "best_model.joblib"
METADATA_PATH: Final[Path] = ARTIFACT_DIR / "model_metadata.json"

TARGET: Final[str] = "Churn"
ID_COL: Final[str] = "customerID"
RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.2

#: Label spellings accepted as the positive (churned) class.
POSITIVE_LABELS: Final[frozenset[str]] = frozenset({"yes", "y", "true", "churn", "1"})
NEGATIVE_LABELS: Final[frozenset[str]] = frozenset({"no", "n", "false", "0"})

#: Optional add-on services; used to derive ``num_addon_services``.
ADDON_SERVICES: Final[list[str]] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

#: Sentinels that really mean "No" (the parent service was not subscribed).
NO_SERVICE_TOKENS: Final[tuple[str, ...]] = ("No internet service", "No phone service")

#: Raw numeric columns that arrive as text in some exports (blank strings).
FORCE_NUMERIC: Final[list[str]] = ["tenure", "MonthlyCharges", "TotalCharges"]

TENURE_BINS: Final[list[float]] = [-0.1, 6, 12, 24, 48, 60, np.inf]
TENURE_LABELS: Final[list[str]] = ["0-6m", "7-12m", "13-24m", "25-48m", "49-60m", "61m+"]

#: Columns consumed by the model after :func:`engineer_features`.
NUMERIC_FEATURES: Final[list[str]] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "spend_trend_ratio",
    "num_addon_services",
]
CATEGORICAL_FEATURES: Final[list[str]] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_bucket",
]

#: Raw columns a caller must supply to ``predict_churn``.
REQUIRED_RAW_COLUMNS: Final[list[str]] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def resolve_data_path(path: str | Path | None = None) -> Path:
    """Locate the raw dataset, accepting either a CSV or an Excel export.

    Args:
        path: Explicit file path. When ``None``, the ``Dataset/`` folder is
            searched for ``Data_file.{xlsx,xls,csv}``.

    Returns:
        Path to an existing data file.

    Raises:
        FileNotFoundError: If no candidate file exists.
    """
    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"Dataset not found: {candidate}")
        return candidate

    for suffix in (".xlsx", ".xls", ".csv"):
        candidate = DATA_DIR / f"Data_file{suffix}"
        if candidate.exists():
            return candidate

    discovered = sorted(DATA_DIR.glob("*.csv")) + sorted(DATA_DIR.glob("*.xls*"))
    if discovered:
        return discovered[0]
    raise FileNotFoundError(f"No CSV/Excel dataset found under {DATA_DIR}")


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw churn dataset with no cleaning applied.

    Blank strings are deliberately preserved so that :func:`clean_frame` stays
    the single, auditable place where data-quality decisions are made.
    """
    resolved = resolve_data_path(path)
    if resolved.suffix.lower() == ".csv":
        return pd.read_csv(resolved)
    return pd.read_excel(resolved)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_frame(
    df: pd.DataFrame,
    *,
    collapse_no_service: bool = True,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Apply deterministic, row-wise cleaning that is safe to run before splitting.

    None of these steps learn a statistic from the data, so running them on the
    full frame cannot leak test information into training.

    Steps:
        1. Trim whitespace on text columns and turn blank strings into ``NaN``
           (``TotalCharges`` ships as ``" "`` for brand-new customers).
        2. Coerce ``tenure`` / ``MonthlyCharges`` / ``TotalCharges`` to numeric.
        3. Set ``TotalCharges`` to ``0.0`` where ``tenure == 0`` -- a customer who
           has not completed a billing cycle has genuinely spent nothing, so the
           median would be badly wrong here.
        4. Render ``SeniorCitizen`` as ``"Yes"``/``"No"`` so it is modelled as the
           categorical flag it really is rather than a magnitude.
        5. Optionally fold ``"No internet service"`` / ``"No phone service"`` into
           ``"No"``; the parent service column already carries that fact, and the
           extra level only adds collinear dummies.
        6. Optionally drop exact duplicate rows.

    Args:
        df: Raw frame as returned by :func:`load_raw_data`.
        collapse_no_service: Collapse the redundant "no parent service" levels.
        drop_duplicates: De-duplicate and reset the index. Correct when building
            a training set; must be ``False`` at inference time, where the caller
            needs exactly one scored row per input row, in the original order.

    Returns:
        A cleaned copy; the input frame is never mutated.
    """
    out = df.copy()

    text_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        stripped = out[col].astype("string").str.strip()
        out[col] = stripped.replace({"": pd.NA}).astype(object)

    for col in FORCE_NUMERIC:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if {"TotalCharges", "tenure"}.issubset(out.columns):
        new_customer = out["TotalCharges"].isna() & out["tenure"].eq(0)
        out.loc[new_customer, "TotalCharges"] = 0.0

    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = (
            out["SeniorCitizen"]
            .map({0: "No", 1: "Yes", "0": "No", "1": "Yes", "No": "No", "Yes": "Yes"})
            .astype(object)
        )

    if collapse_no_service:
        service_cols = [c for c in ADDON_SERVICES + ["MultipleLines"] if c in out.columns]
        replacement = dict.fromkeys(NO_SERVICE_TOKENS, "No")
        for col in service_cols:
            out[col] = out[col].replace(replacement)

    if drop_duplicates:
        return out.drop_duplicates().reset_index(drop=True)
    return out


def encode_target(series: pd.Series) -> pd.Series:
    """Map a Yes/No (or 1/0) churn column to an ``int8`` 0/1 series.

    Raises:
        ValueError: If any value cannot be interpreted as a binary label.
    """
    normalised = series.astype("string").str.strip().str.lower()
    unknown = normalised.dropna()[~normalised.dropna().isin(POSITIVE_LABELS | NEGATIVE_LABELS)]
    if not unknown.empty:
        raise ValueError(f"Unrecognised {TARGET} labels: {sorted(set(unknown))[:5]}")
    return normalised.isin(POSITIVE_LABELS).astype("int8")


# --------------------------------------------------------------------------- #
# Feature engineering -- runs inside the pipeline, so serving matches training
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive business-meaningful features from the raw columns.

    Every derivation is row-local (no group statistics, no target lookups), which
    is what makes it safe to execute inside the pipeline on a single unseen row.

    Adds:
        ``tenure_bucket``: lifecycle stage, the way retention teams segment.
        ``avg_monthly_spend``: lifetime spend per active month.
        ``spend_trend_ratio``: current bill vs. lifetime average -- above ``1``
            flags a customer whose price recently rose, a classic churn trigger.
        ``num_addon_services``: count of optional services held; a stickiness proxy.

    Args:
        df: Frame containing the raw Telco columns.

    Returns:
        A copy with the derived columns appended.
    """
    out = df.copy()

    tenure = pd.to_numeric(out["tenure"], errors="coerce")
    monthly = pd.to_numeric(out["MonthlyCharges"], errors="coerce")
    total = pd.to_numeric(out["TotalCharges"], errors="coerce")

    out["tenure_bucket"] = pd.cut(
        tenure, bins=TENURE_BINS, labels=TENURE_LABELS, right=True
    ).astype(object)

    # Guard the first billing cycle: tenure 0 would divide by zero.
    out["avg_monthly_spend"] = total / tenure.clip(lower=1)
    out["spend_trend_ratio"] = monthly / out["avg_monthly_spend"].replace(0, np.nan)

    present_addons = [c for c in ADDON_SERVICES if c in out.columns]
    if present_addons:
        held = out[present_addons].apply(lambda s: s.astype("string").str.strip().eq("Yes"))
        out["num_addon_services"] = held.sum(axis=1).astype("float64")
    else:
        out["num_addon_services"] = 0.0

    derived = ["avg_monthly_spend", "spend_trend_ratio", "num_addon_services"]
    out[derived] = out[derived].replace([np.inf, -np.inf], np.nan)
    return out


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Column-wise preprocessing: median-impute + scale, mode-impute + one-hot.

    Every statistic (medians, modes, category levels, scaler means and standard
    deviations) is learned during ``fit`` only. Keeping these inside the pipeline
    is what guarantees the test set -- and later, production traffic -- stays
    genuinely unseen.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator: Any) -> Pipeline:
    """Wrap an estimator into the full raw-DataFrame-in, prediction-out pipeline.

    Args:
        estimator: Any scikit-learn compatible classifier.

    Returns:
        ``Pipeline`` of feature engineering -> preprocessing -> classifier.
    """
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(engineer_features, validate=False)),
            ("preprocess", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def get_encoded_feature_names(pipeline: Pipeline) -> list[str]:
    """Return post-one-hot feature names from a fitted pipeline."""
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split_data(
    df: pd.DataFrame,
    *,
    target: str = TARGET,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a cleaned frame into stratified train/test sets.

    The identifier column is dropped so the model can never key off it.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    y = encode_target(df[target])
    drop_cols = [c for c in (target, ID_COL) if c in df.columns]
    X = df.drop(columns=drop_cols)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def ensure_directories() -> None:
    """Create the artifact and figure output folders if they do not exist."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
