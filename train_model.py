"""
train_model.py
--------------
CLI script: trains the readmission model and saves artifacts to model/.

Run: python train_model.py

All logic lives in model_utils.py so it can be reused by the Streamlit app.
"""

from model_utils import DEFAULT_DATA_PATH, DEFAULT_MODEL_DIR, train_and_save_model


def main():
    print("Loading data …")
    print("Training Random Forest …")

    pipeline, feature_names, metrics = train_and_save_model(
        data_path=DEFAULT_DATA_PATH,
        model_dir=DEFAULT_MODEL_DIR,
    )

    print("\n── Test-Set Performance ──────────────────────────────────")
    print(f"  Accuracy : {metrics['accuracy']:.3f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.3f}")
    print(metrics["report"])


if __name__ == "__main__":
    main()
