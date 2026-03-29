"""
model_utils.py
--------------
Reusable training and loading utilities for the CareReturn AI readmission model.

Public API
----------
train_and_save_model(data_path, model_dir)  -> (pipeline, feature_names, metrics)
load_or_train_model(data_path, model_dir)   -> (pipeline, feature_names, metrics, was_retrained)
"""

import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── feature schema ────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "age",
    "prior_admissions",
    "comorbidity_count",
    "num_medications",
    "abnormal_lab_flag",
    "length_of_stay",
    "followup_scheduled",
]
CATEGORICAL_FEATURES = ["admission_type", "discharge_destination"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "readmitted_30d"

# ── default paths ─────────────────────────────────────────────────────────────

DEFAULT_DATA_PATH = "data/patients.csv"
DEFAULT_MODEL_DIR = "model"


# ── internal helpers ──────────────────────────────────────────────────────────

def _build_pipeline() -> Pipeline:
    """Return an unfitted sklearn Pipeline.

    Uses only standard sklearn transformers — no lambdas, no custom objects —
    so the fitted pipeline serialises cleanly with joblib across environments.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def _get_feature_names(pipeline: Pipeline) -> list:
    """Return ordered feature names after the pipeline has been fitted."""
    cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERIC_FEATURES + cat_names


def _ensure_data(data_path: str) -> None:
    """Generate the synthetic dataset if it does not already exist."""
    if os.path.exists(data_path):
        return
    # Import here so generate_data.py remains a standalone script as well.
    from generate_data import generate_dataset

    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    df = generate_dataset()
    df.to_csv(data_path, index=False)
    print(f"[model_utils] Dataset generated → {data_path}")


# ── public API ────────────────────────────────────────────────────────────────

def train_and_save_model(
    data_path: str = DEFAULT_DATA_PATH,
    model_dir: str = DEFAULT_MODEL_DIR,
) -> tuple:
    """Train the readmission model and persist artifacts to *model_dir*.

    Parameters
    ----------
    data_path : str
        Path to the CSV dataset. Generated automatically if absent.
    model_dir : str
        Directory where joblib artifacts are written.

    Returns
    -------
    (pipeline, feature_names, metrics) tuple
    """
    _ensure_data(data_path)
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "report": classification_report(y_test, y_pred),
    }

    feature_names = _get_feature_names(pipeline)

    joblib.dump(pipeline, os.path.join(model_dir, "readmission_model.joblib"))
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.joblib"))
    joblib.dump(metrics, os.path.join(model_dir, "metrics.joblib"))
    print(f"[model_utils] Artifacts saved to '{model_dir}/'")

    return pipeline, feature_names, metrics


def load_or_train_model(
    data_path: str = DEFAULT_DATA_PATH,
    model_dir: str = DEFAULT_MODEL_DIR,
) -> tuple:
    """Load saved model artifacts, retraining from scratch if loading fails.

    This makes the app portable: on Streamlit Community Cloud the model
    directory may be absent or contain artifacts built on a different
    scikit-learn version. In both cases the fallback retrains cleanly.

    Returns
    -------
    (pipeline, feature_names, metrics, was_retrained)
        was_retrained is True when the fallback path was taken.
    """
    model_path = os.path.join(model_dir, "readmission_model.joblib")
    feature_path = os.path.join(model_dir, "feature_names.joblib")
    metrics_path = os.path.join(model_dir, "metrics.joblib")

    try:
        pipeline = joblib.load(model_path)
        feature_names = joblib.load(feature_path)
        metrics = joblib.load(metrics_path)
        return pipeline, feature_names, metrics, False
    except Exception as exc:
        print(f"[model_utils] Load failed ({exc}). Retraining …")
        pipeline, feature_names, metrics = train_and_save_model(data_path, model_dir)
        return pipeline, feature_names, metrics, True
