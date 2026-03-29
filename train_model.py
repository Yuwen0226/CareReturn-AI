"""
train_model.py
--------------
Trains a Random Forest classifier on the synthetic patient dataset and saves:
  - model/readmission_model.joblib   (trained pipeline)
  - model/feature_names.joblib       (ordered feature names)
  - model/metrics.joblib             (test-set performance metrics)

Run: python train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = "data/patients.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── feature definitions ───────────────────────────────────────────────────────
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
TARGET = "readmitted_30d"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Run generate_data.py first."
        )
    return pd.read_csv(path)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
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


def get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    cat_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return NUMERIC_FEATURES + cat_names


def main():
    print("Loading data …")
    df = load_data(DATA_PATH)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Training Random Forest …")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "report": classification_report(y_test, y_pred),
    }

    print("\n── Test-Set Performance ──────────────────────────────────")
    print(f"  Accuracy : {metrics['accuracy']:.3f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.3f}")
    print(metrics["report"])

    feature_names = get_feature_names(pipeline)

    joblib.dump(pipeline, f"{MODEL_DIR}/readmission_model.joblib")
    joblib.dump(feature_names, f"{MODEL_DIR}/feature_names.joblib")
    joblib.dump(metrics, f"{MODEL_DIR}/metrics.joblib")
    print(f"Model artifacts saved to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
