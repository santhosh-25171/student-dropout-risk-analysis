"""
Page 8: Comprehensive Model Evaluation, Fairness, and Limitations.
Cross-paradigm ML vs DL comparison, demographic subgroup fairness, and transparent limitations.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path

from src.config import DL_RESULTS_DIR, ML_RESULTS_DIR, PRIMARY_DATASET_NAME
from src.data.loader import load_dataset
from src.data.validation import analyze_subgroups
from src.ui_components import apply_custom_theme, render_header

st.set_page_config(page_title="Model Evaluation — EDUVISION AI", page_icon="⚖️", layout="wide")
apply_custom_theme()

render_header(
    title="⚖️ Comprehensive Evaluation, Fairness Audit & Limitations",
    subtitle="Holistic benchmarking across Machine Learning and Deep Learning, subgroup fairness, and ethics"
)

tab1, tab2, tab3 = st.tabs(["📊 Cross-Paradigm Benchmark (ML vs DL)", "🤝 Subgroup Fairness Audit", "⚠️ Institutional Limitations & Ethics"])

with tab1:
    st.subheader("Holdout Test Set Benchmark (Identical 15% Untouched Partition)")
    
    comp_file = DL_RESULTS_DIR / "primary_ml_vs_dl_comparison.csv"
    comp_img = DL_RESULTS_DIR / "primary_ml_vs_dl_comparison.png"

    if comp_file.exists():
        df_bench = pd.read_csv(comp_file)
        st.dataframe(
            df_bench.style.highlight_max(subset=["Accuracy", "Recall", "F1_Score", "ROC_AUC"], color="#dcfce7"),
            use_container_width=True
        )

    if comp_img.exists():
        st.image(str(comp_img), caption="Comparative Metric Performance across Paradigms", use_container_width=True)

    st.markdown(
        """
        > **Evaluation Analysis:**
        > - Both **ANN (F1: 86.73%, Acc: 90.00%)** and **DNN (Recall: 92.59%, Acc: 89.33%)** perform strongly on tabular student data.
        > - **Logistic Regression (Recall: 88.89%, ROC-AUC: 0.9655)** remains an exceptionally competitive, parsimonious baseline with direct linear interpretability.
        > - In educational retention settings, **Recall (Sensitivity)** must be the primary objective metric: failing to detect an at-risk student is significantly more detrimental than issuing an unwarranted advisory check-in.
        """
    )

with tab2:
    st.subheader("Demographic Subgroup Fairness & Disparity Audit")
    st.markdown("Auditing retention and dropout distributions across demographic cohorts to verify ethical parity:")

    df_raw = load_dataset(PRIMARY_DATASET_NAME)

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.markdown("#### Gender Subgroup Distribution")
        gender_fairness = analyze_subgroups(df_raw, "gender")
        st.dataframe(gender_fairness, use_container_width=True)

    with fcol2:
        st.markdown("#### Socioeconomic Tier Distribution")
        ses_fairness = analyze_subgroups(df_raw, "socioeconomic_status")
        st.dataframe(ses_fairness, use_container_width=True)

    st.info(
        "💡 **Fairness Principle:** Protected attributes (gender, socioeconomic background) should never serve as automated negative scoring mechanisms. "
        "Any interventions triggered for lower socioeconomic tiers must focus on equitable resource allocation (e.g. food security, fee bursaries, transport stipends)."
    )

with tab3:
    st.subheader("Academic Limitations and Real-World Considerations")
    st.markdown(
        """
        ### 1. Dataset Sample Size and Cohort Scope
        - The primary dataset includes 1,000 student records. While sufficient for statistical power in cross-validation, larger institutional datasets ($>10,000$ students) are necessary for capturing rare corner cases and multi-year systemic shifts.
        
        ### 2. Dataset Synthesis vs Observational Data
        - In the secondary dataset (`university_student_retention_dataset_2134.csv`), our data audit proved that background demographic covariates had near-zero natural correlation to dropout, while variables like `dropout_risk` and `dropout_cause` were post-hoc outcome tags. Excluded to prevent severe target leakage, models trained on those uninformative covariates correctly achieved near-chance accuracy (~52%), demonstrating rigorous data hygiene.
        
        ### 3. Human-in-the-Loop Imperative
        - Model predictions represent **probabilistic alerts**, not deterministic student fates. Quantitative predictions must always be contextualized by professional academic advisors considering qualitative life circumstances (family emergencies, health challenges, career shifts).
        
        ### 4. Dynamic Temporal Drift
        - Curriculum changes, grading policy adjustments, or economic fluctuations can induce concept drift over semesters. Preprocessing and modeling pipelines must be re-calibrated annually.
        """
    )
