"""
Page 7: Explainable AI (XAI) & SHAP Interpretability.
Provides global feature attribution, beeswarm summary, and individual waterfall attributions.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from src.config import EXPLAIN_RESULTS_DIR
from src.explainability.shap_analysis import compute_global_shap, explain_single_prediction
from src.ui_components import apply_custom_theme, render_header

st.set_page_config(page_title="Explainable AI — EDUVISION AI", page_icon="💡", layout="wide")
apply_custom_theme()

render_header(
    title="💡 Explainable AI (XAI) — Algorithmic Transparency with SHAP",
    subtitle="Interpretable machine learning attributions based on Shapley additive values"
)

st.markdown(
    """
    <div class="callout-info">
        <strong>Why Explainability Matters in Education:</strong><br>
        Machine learning models can identify complex multi-dimensional risk patterns, but educators cannot act on "black box" scores.
        Using <strong>SHAP (SHapley Additive exPlanations)</strong>, every single prediction is decomposed into exact feature contributions, 
        showing advisors <em>why</em> a student is flagged and <em>what</em> actionable levers can be pulled.
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["🌐 Global Feature Importance", "🔬 Individual Prediction Breakdown"])

with tab1:
    st.subheader("Global Feature Attribution Across Student Cohort")
    
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        model_pick = st.selectbox(
            "Select Model Architecture for Global XAI:",
            ["Random_Forest", "Logistic_Regression", "Gradient_Boosting"],
            format_func=lambda x: x.replace("_", " ")
        )

    # Check if pre-computed chart exists or compute
    beeswarm_file = EXPLAIN_RESULTS_DIR / f"primary_{model_pick}_shap_summary.png"
    bar_file = EXPLAIN_RESULTS_DIR / f"primary_{model_pick}_shap_bar.png"

    if not beeswarm_file.exists() or not bar_file.exists():
        with st.spinner(f"Computing SHAP values for {model_pick.replace('_', ' ')}..."):
            compute_global_shap(model_pick, "primary", max_samples=75)

    gcol1, gcol2 = st.columns(2)
    with gcol1:
        if beeswarm_file.exists():
            st.image(str(beeswarm_file), caption="SHAP Summary Beeswarm Plot (Directionality & Density)", use_container_width=True)

    with gcol2:
        if bar_file.exists():
            st.image(str(bar_file), caption="Mean Absolute SHAP Values (Global Magnitude)", use_container_width=True)

    st.markdown("---")
    st.subheader("📖 How to Read SHAP Global Plots")
    st.markdown(
        """
        - **Beeswarm Plot:** Each dot represents an individual student. Red points represent high feature values (e.g. high commute or high part-time hours); blue dots represent low feature values. Dots to the **right of zero** increase dropout odds; dots to the **left of zero** protect retention.
        - **Bar Plot:** Ranks features purely by overall impact on model output across the student body.
        """
    )

with tab2:
    st.subheader("Simulate Individual Student Attribution")
    st.markdown("Adjust parameters below to generate a live SHAP waterfall plot explaining the model's decision logic:")

    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        sim_gpa = st.slider("Simulated GPA:", 1.0, 4.0, 2.1, 0.1)
        sim_att = st.slider("Simulated Attendance (%):", 50.0, 100.0, 68.0, 1.0)
    with scol2:
        sim_aid = st.selectbox("Financial Aid:", [1, 0], format_func=lambda x: "Yes (Receiving Aid)" if x == 1 else "No Aid")
        sim_job = st.selectbox("Part-time Job:", [0, 1], format_func=lambda x: "No Job" if x == 0 else "Working Part-time")
    with scol3:
        sim_ses = st.selectbox("Socioeconomic Status:", ["Low", "Medium", "High"], index=0)
        sim_commute = st.slider("Commute Distance (km):", 1.0, 50.0, 28.0, 1.0)

    sim_student = {
        "age": 20,
        "gender": "Female",
        "socioeconomic_status": sim_ses,
        "commute_distance": float(sim_commute),
        "attendance_rate": float(sim_att),
        "previous_grades": 65.0,
        "current_gpa": float(sim_gpa),
        "financial_aid": sim_aid,
        "part_time_job": sim_job,
        "extracurricular_activities": 0
    }

    if st.button("Generate SHAP Waterfall Attribution"):
        with st.spinner("Calculating Shapley values for simulated student..."):
            expl = explain_single_prediction(sim_student, model_name=model_pick, dataset_prefix="primary")
            
            pcol1, pcol2 = st.columns([1, 1])
            with pcol1:
                st.image(expl["waterfall_path"], caption="Waterfall Attribution (Log-Odds Impact)", use_container_width=True)
            with pcol2:
                st.markdown("#### Natural Language Advisor Briefing")
                for n in expl["narrative_explanations"]:
                    st.markdown(f"- 📢 {n}")
                
                st.markdown("#### Tabular Feature Contributions")
                st.dataframe(pd.DataFrame(expl["all_contributions"]), use_container_width=True)
