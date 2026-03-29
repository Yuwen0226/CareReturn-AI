"""
generate_data.py
----------------
Creates a synthetic patient dataset for 30-day hospital readmission prediction.
Run this once to produce data/patients.csv used by train_model.py.
"""

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_PATIENTS = 2000

np.random.seed(RANDOM_SEED)

os.makedirs("data", exist_ok=True)


def generate_dataset(n: int = N_PATIENTS) -> pd.DataFrame:
    age = np.random.randint(18, 90, size=n)
    prior_admissions = np.random.poisson(lam=1.5, size=n).clip(0, 10)
    comorbidity_count = np.random.poisson(lam=2.0, size=n).clip(0, 8)
    num_medications = np.random.randint(1, 20, size=n)
    abnormal_lab_flag = np.random.choice([0, 1], size=n, p=[0.55, 0.45])
    length_of_stay = np.random.gamma(shape=2.5, scale=2.0, size=n).clip(1, 30).astype(int)

    admission_type = np.random.choice(
        ["Emergency", "Elective", "Urgent"], size=n, p=[0.50, 0.30, 0.20]
    )
    discharge_destination = np.random.choice(
        ["Home", "SNF", "Home Health", "Rehab"], size=n, p=[0.55, 0.20, 0.15, 0.10]
    )
    followup_scheduled = np.random.choice([0, 1], size=n, p=[0.35, 0.65])

    # ---------- label construction (domain-informed log-odds) ----------
    log_odds = (
        -2.5
        + 0.02 * (age - 50)
        + 0.35 * prior_admissions
        + 0.25 * comorbidity_count
        + 0.04 * num_medications
        + 0.50 * abnormal_lab_flag
        + 0.08 * length_of_stay
        + 0.60 * (admission_type == "Emergency").astype(int)
        + 0.45 * (discharge_destination == "SNF").astype(int)
        - 0.55 * followup_scheduled
    )
    prob = 1 / (1 + np.exp(-log_odds))
    readmitted = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "prior_admissions": prior_admissions,
            "comorbidity_count": comorbidity_count,
            "num_medications": num_medications,
            "abnormal_lab_flag": abnormal_lab_flag,
            "length_of_stay": length_of_stay,
            "admission_type": admission_type,
            "discharge_destination": discharge_destination,
            "followup_scheduled": followup_scheduled,
            "readmitted_30d": readmitted,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "data/patients.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset saved → {out_path}  ({len(df)} rows, {df['readmitted_30d'].mean():.1%} readmission rate)")
