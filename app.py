"""
app.py
------
CareReturn AI – 30-Day Hospital Readmission Risk Predictor
Streamlit web application.

Run: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from model_utils import DEFAULT_DATA_PATH, DEFAULT_MODEL_DIR, load_or_train_model

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareReturn AI",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── constants ─────────────────────────────────────────────────────────────────
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

FEATURE_LABELS = {
    "age": "Age",
    "prior_admissions": "Prior admissions (past 12 months)",
    "comorbidity_count": "Number of comorbidities",
    "num_medications": "Number of current medications",
    "abnormal_lab_flag": "Abnormal lab results at discharge",
    "length_of_stay": "Length of stay (days)",
    "followup_scheduled": "Follow-up appointment scheduled",
    "admission_type_Emergency": "Admission type: Emergency",
    "admission_type_Elective": "Admission type: Elective",
    "admission_type_Urgent": "Admission type: Urgent",
    "discharge_destination_Home": "Discharge to: Home",
    "discharge_destination_SNF": "Discharge to: Skilled Nursing Facility",
    "discharge_destination_Home Health": "Discharge to: Home Health",
    "discharge_destination_Rehab": "Discharge to: Rehab",
}

RISK_ACTIONS = {
    "Low": {
        "summary": "Standard discharge workflow. No elevated intervention required.",
        "actions": [
            "Complete standard discharge checklist and provide written after-visit summary.",
            "Schedule a routine outpatient follow-up within 2–4 weeks.",
            "Confirm the patient can verbalize their diagnosis, medications, and any activity restrictions.",
            "Provide written guidance on symptoms that should prompt the patient to seek care.",
            "Ensure the patient has an adequate supply of all prescribed medications at discharge.",
        ],
    },
    "Medium": {
        "summary": "Moderate risk. Proactive outreach and closer monitoring are advised.",
        "actions": [
            "Schedule a follow-up appointment within 7–14 days before the patient leaves the unit.",
            "Conduct a post-discharge phone check-in within 48 hours to assess recovery and medication adherence.",
            "Complete a full medication reconciliation and resolve any discrepancies prior to discharge.",
            "Educate the patient and caregiver on early warning signs that require immediate attention.",
            "Coordinate with the primary care physician to ensure continuity of care.",
            "Assess and address any social or logistical barriers to attending the follow-up appointment.",
        ],
    },
    "High": {
        "summary": "Elevated risk. Intensive transitional care measures are strongly recommended.",
        "actions": [
            "Escalate to a care manager or social worker for a discharge planning review before the patient leaves.",
            "Schedule an urgent follow-up appointment within 48–72 hours of discharge.",
            "Perform a structured medication reconciliation and verify the patient understands all changes.",
            "Arrange a home health or transitional care nurse visit within 24–48 hours post-discharge.",
            "Initiate a formal Transitional Care Management (TCM) program and assign a care coordinator.",
            "Conduct a 24-hour post-discharge phone call; document findings and escalate if condition has declined.",
            "Review and confirm all specialist follow-up appointments are booked before discharge.",
            "Screen for social determinants of health (transportation, housing, caregiver support) and refer as needed.",
        ],
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_model():
    """Load the saved model, retraining automatically if the artifact is
    missing or was built with an incompatible scikit-learn version."""
    return load_or_train_model(
        data_path=DEFAULT_DATA_PATH,
        model_dir=DEFAULT_MODEL_DIR,
    )


def risk_level(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    elif prob < 0.60:
        return "Medium"
    return "High"


def risk_color(level: str) -> str:
    return {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}[level]


def top_factors(pipeline, feature_names: list[str], input_df: pd.DataFrame, top_n: int = 5):
    """Return top_n features by absolute importance contribution for this patient."""
    rf = pipeline.named_steps["classifier"]
    transformed = pipeline.named_steps["preprocessor"].transform(input_df)
    importances = rf.feature_importances_
    contributions = np.abs(transformed[0] * importances)
    idx = np.argsort(contributions)[::-1][:top_n]
    results = []
    for i in idx:
        fname = feature_names[i]
        label = FEATURE_LABELS.get(fname, fname)
        val = transformed[0][i]
        direction = "increases" if val * importances[i] > 0 else "decreases"
        results.append((label, importances[i], direction))
    return results


def build_input_df(inputs: dict) -> pd.DataFrame:
    row = {k: inputs[k] for k in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    return pd.DataFrame([row])


# ── UI ────────────────────────────────────────────────────────────────────────

def render_header():
    st.markdown(
        """
        <div style='text-align:center; padding: 1.5rem 0 0.5rem'>
            <h1 style='font-size:2.4rem; font-weight:700; color:#1a3c5e;'>
                🏥 CareReturn AI
            </h1>
            <p style='font-size:1.1rem; color:#4a6785;'>
                30-Day Hospital Readmission Risk Predictor
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


def render_disclaimer():
    st.info(
        "**Clinical Disclaimer:** CareReturn AI is a decision-support tool intended to assist—"
        " not replace—the clinical judgment of licensed healthcare professionals. "
        "Predictions are based on statistical patterns and should always be interpreted "
        "in the context of a patient's full clinical picture.",
        icon="⚠️",
    )


def render_model_metrics(metrics: dict):
    with st.expander("Model performance (test set)", expanded=False):
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        c2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        st.caption("Trained on synthetic data — not validated for clinical use.")


def render_input_form() -> dict:
    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=110, value=65, step=1)
        prior_admissions = st.number_input(
            "Prior admissions (past 12 months)", min_value=0, max_value=20, value=1, step=1
        )
        comorbidity_count = st.number_input(
            "Number of comorbidities", min_value=0, max_value=15, value=2, step=1
        )
        num_medications = st.number_input(
            "Number of current medications", min_value=0, max_value=40, value=5, step=1
        )

    with col2:
        length_of_stay = st.number_input(
            "Length of stay (days)", min_value=1, max_value=60, value=4, step=1
        )
        admission_type = st.selectbox(
            "Admission type", options=["Emergency", "Elective", "Urgent"]
        )
        discharge_destination = st.selectbox(
            "Discharge destination",
            options=["Home", "SNF", "Home Health", "Rehab"],
            format_func=lambda x: {
                "Home": "Home",
                "SNF": "Skilled Nursing Facility (SNF)",
                "Home Health": "Home Health",
                "Rehab": "Rehabilitation Facility",
            }[x],
        )

    st.markdown("")
    col3, col4 = st.columns(2)
    with col3:
        abnormal_lab_flag = st.checkbox("Abnormal lab results at discharge", value=False)
    with col4:
        followup_scheduled = st.checkbox("Follow-up appointment scheduled", value=True)

    return {
        "age": age,
        "prior_admissions": prior_admissions,
        "comorbidity_count": comorbidity_count,
        "num_medications": num_medications,
        "abnormal_lab_flag": int(abnormal_lab_flag),
        "length_of_stay": length_of_stay,
        "admission_type": admission_type,
        "discharge_destination": discharge_destination,
        "followup_scheduled": int(followup_scheduled),
    }


def render_results(pipeline, feature_names, inputs: dict):
    input_df = build_input_df(inputs)
    prob = pipeline.predict_proba(input_df)[0][1]
    level = risk_level(prob)
    color = risk_color(level)

    st.divider()
    st.subheader("Prediction Results")

    # ── risk banner ───────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='
            background:{color}22;
            border-left: 6px solid {color};
            border-radius: 6px;
            padding: 1rem 1.4rem;
            margin-bottom: 1rem;
        '>
            <span style='font-size:1.05rem; color:#333;'>
                Readmission Risk Probability
            </span><br>
            <span style='font-size:2.8rem; font-weight:700; color:{color};'>
                {prob:.1%}
            </span>
            <span style='
                font-size:1.1rem;
                font-weight:600;
                color:{color};
                margin-left:0.8rem;
                background:{color}33;
                padding:0.2rem 0.8rem;
                border-radius:20px;
            '>
                {level} Risk
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── probability bar ───────────────────────────────────────────────────────
    st.progress(int(prob * 100))
    st.caption("0% — Low risk ←————————→ 100% — High risk")

    # ── top factors ───────────────────────────────────────────────────────────
    st.markdown("#### Top Factors Influencing This Prediction")
    factors = top_factors(pipeline, feature_names, input_df)
    for label, importance, _ in factors:
        bar_pct = int(importance * 1000)
        bar_pct = min(bar_pct, 100)
        st.markdown(
            f"""
            <div style='margin-bottom:0.5rem;'>
                <span style='font-size:0.95rem;'>{label}</span>
                <div style='background:#e8edf2; border-radius:4px; height:10px; margin-top:3px;'>
                    <div style='
                        width:{bar_pct}%;
                        background:#1a3c5e;
                        height:10px;
                        border-radius:4px;
                    '></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── recommended actions ───────────────────────────────────────────────────
    st.markdown("#### Recommended Next Actions")
    action_data = RISK_ACTIONS[level]
    st.markdown(
        f"<p style='color:{color}; font-weight:600; margin-bottom:0.6rem;'>"
        f"{action_data['summary']}</p>",
        unsafe_allow_html=True,
    )
    for action in action_data["actions"]:
        st.markdown(f"- {action}")

    st.divider()
    render_disclaimer()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    render_header()

    with st.spinner("Loading model …"):
        pipeline, feature_names, metrics, was_retrained = get_model()

    if was_retrained:
        st.toast("Model retrained from scratch (no saved artifact found).", icon="ℹ️")

    render_model_metrics(metrics)
    render_disclaimer()

    st.markdown("")
    inputs = render_input_form()

    st.markdown("")
    predict_btn = st.button("Predict Readmission Risk", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("Analyzing patient profile …"):
            render_results(pipeline, feature_names, inputs)

    st.markdown(
        "<br><p style='text-align:center; color:#aaa; font-size:0.8rem;'>"
        "CareReturn AI · Built for educational purposes · Not for clinical use"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
