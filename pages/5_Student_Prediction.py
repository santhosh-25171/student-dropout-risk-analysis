"""
Page 5: Individual Student Dropout Risk Prediction.
Interactive input form, model selection, calibrated probability, risk classification, and recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import DEFAULT_RISK_THRESHOLDS
from src.ml.predict import predict_single_student, assign_risk_level
from src.explainability.shap_analysis import explain_single_prediction
from src.ui_components import apply_custom_theme, render_header, render_risk_badge, render_ethical_notice

st.set_page_config(page_title="Student Prediction — EDUVISION AI", page_icon="🎯", layout="wide")
apply_custom_theme()

render_header(
    title="🎯 Individual Student Risk Assessment & Intervention Planner",
    subtitle="Simulate and evaluate dropout vulnerability for individual student profiles"
)

render_ethical_notice()

# Sidebar: Configuration & Risk Thresholds
st.sidebar.subheader("⚙️ Decision Settings")
model_options = [
    "Logistic_Regression",
    "Random_Forest",
    "Gradient_Boosting",
    "Support_Vector_Machine",
    "Decision_Tree"
]
selected_model = st.sidebar.selectbox(
    "Select Predictive Model:",
    model_options,
    format_func=lambda x: x.replace("_", " ")
)

st.sidebar.markdown("#### Configurable Risk Thresholds")
low_thresh = st.sidebar.slider("Low/Moderate Threshold:", 0.10, 0.45, DEFAULT_RISK_THRESHOLDS["low_max"], step=0.05)
high_thresh = st.sidebar.slider("Moderate/High Threshold:", 0.50, 0.85, DEFAULT_RISK_THRESHOLDS["medium_max"], step=0.05)
custom_thresholds = {"low_max": low_thresh, "medium_max": high_thresh}

# Student Input Form
st.subheader("📝 Enter Student Academic & Demographic Profile")

with st.form("student_profile_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age:", min_value=17, max_value=35, value=20, step=1)
        gender = st.selectbox("Gender:", ["Male", "Female", "Other"])
        socioeconomic = st.selectbox("Socioeconomic Status:", ["Low", "Medium", "High"], index=1)
        commute = st.slider("Commute Distance (km):", 1.0, 50.0, 15.0, step=0.5)

    with col2:
        attendance = st.slider("Attendance Rate (%):", 40.0, 100.0, 82.0, step=1.0)
        current_gpa = st.slider("Current Cumulative GPA (4.0 Scale):", 1.0, 4.0, 2.85, step=0.05)
        prev_grades = st.slider("Previous Semester Grades (%):", 30.0, 100.0, 75.0, step=1.0)

    with col3:
        financial_aid = st.radio("Financial Aid Recipient?", ["Yes (Receiving Aid)", "No Aid"], index=0)
        part_time_job = st.radio("Part-Time Employment?", ["No Job", "Working Part-Time"], index=0)
        extracurricular = st.radio("Extracurricular Participation?", ["Active Participant", "No Participation"], index=0)

    submit_btn = st.form_submit_button("🔍 Compute Student Risk Assessment", use_container_width=True)

if submit_btn:
    # Prepare student dictionary
    student_dict = {
        "age": int(age),
        "gender": gender,
        "socioeconomic_status": socioeconomic,
        "commute_distance": float(commute),
        "attendance_rate": float(attendance),
        "previous_grades": float(prev_grades),
        "current_gpa": float(current_gpa),
        "financial_aid": 1 if "Yes" in financial_aid else 0,
        "part_time_job": 1 if "Working" in part_time_job else 0,
        "extracurricular_activities": 1 if "Active" in extracurricular else 0
    }

    with st.spinner("Analyzing risk factors and generating model inference..."):
        try:
            pred_res = predict_single_student(
                student_dict,
                model_name=selected_model,
                dataset_prefix="primary",
                thresholds=custom_thresholds
            )

            st.markdown("---")
            st.subheader("📋 Assessment Results")

            rcol1, rcol2, rcol3 = st.columns(3)
            with rcol1:
                st.metric("Predicted Outcome", pred_res["prediction_label"])
            with rcol2:
                st.metric("Estimated Dropout Probability", f"{pred_res['dropout_probability_pct']}%")
            with rcol3:
                badge_html = render_risk_badge(pred_res["risk_level"])
                st.markdown(f"**Classification:**<br>{badge_html}", unsafe_allow_html=True)

            # Recommendations
            st.markdown("#### 🎯 Prioritized Early Intervention Actions")
            for action in pred_res["recommendations"]:
                st.markdown(f"- 📌 **Action Item:** {action}")

            # Local Explainability
            st.markdown("---")
            st.subheader("💡 Key Contributing Factors (SHAP Explanation)")
            
            try:
                shap_res = explain_single_prediction(student_dict, model_name=selected_model, dataset_prefix="primary")
                
                ecol1, ecol2 = st.columns([1, 1])
                with ecol1:
                    st.markdown("##### 🚨 Top Factors Increasing Risk:")
                    for factor in shap_res["top_risk_factors"]:
                        st.markdown(f"- **{factor['feature'].replace('_', ' ').title()}:** Added `+{factor['shap_value']:.2f}` risk weight.")
                    
                    st.markdown("##### 🛡️ Top Factors Protecting Retention:")
                    for factor in shap_res["top_protective_factors"]:
                        st.markdown(f"- **{factor['feature'].replace('_', ' ').title()}:** Reduced `-{abs(factor['shap_value']):.2f}` risk weight.")

                with ecol2:
                    if Path(shap_res["waterfall_path"]).exists():
                        st.image(shap_res["waterfall_path"], caption="Local SHAP Attribution Breakdown", use_container_width=True)

                st.markdown("##### 🗣️ plain-English Academic Interpretation:")
                for narr in shap_res["narrative_explanations"]:
                    st.markdown(f"> {narr}")

            except Exception as e:
                st.info(f"Local SHAP attribution explanation note: {e}")

        except Exception as err:
            st.error(f"Inference error: {err}")
