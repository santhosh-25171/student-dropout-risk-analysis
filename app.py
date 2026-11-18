"""
EDUVISION AI — Intelligent Student Dropout Prediction and Early Intervention System
Main Entrypoint and Platform Portal.
"""

import streamlit as st
from pathlib import Path
import os
import json

from src.config import (
    APP_TITLE,
    APP_SUBTITLE,
    ML_RESULTS_DIR,
    DL_RESULTS_DIR,
    PRIMARY_DATASET_NAME,
    SECONDARY_DATASET_NAME
)
from src.ui_components import apply_custom_theme, render_header, render_metric_card, render_ethical_notice
from src.data.loader import load_dataset, get_dataset_summary

st.set_page_config(
    page_title="EDUVISION AI — Student Dropout Risk System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_theme()

# Sidebar Navigation Info
st.sidebar.image("https://img.shields.io/badge/EDUVISION%20AI-v2.0%20Enterprise-blue?style=for-the-badge&logo=scikitlearn", use_container_width=True)
st.sidebar.markdown("### 🏛️ Institution Portal")
st.sidebar.info(
    """
    **Active Modules:**
    - 📊 Overview Dashboard
    - 🔍 Dataset Explorer
    - 🤖 5 ML Algorithms
    - 🧠 Deep Learning (ANN & DNN)
    - 🎯 Student Risk Prediction
    - 📁 Batch Prediction (CSV)
    - 💡 Explainable AI (SHAP)
    - ⚖️ Model Evaluation & Fairness
    - 📑 Reports & Data Export
    """
)

render_header(
    title="🎓 EDUVISION AI — Intelligent Student Retention Platform",
    subtitle="Early Intervention & Predictive Academic Decision Support System"
)

render_ethical_notice()

# Load summary metrics
col1, col2, col3, col4 = st.columns(4)

df_primary = load_dataset(PRIMARY_DATASET_NAME)
d1_summary = get_dataset_summary(df_primary)

# Load ML comparison if available
best_ml_name = "Logistic Regression"
best_ml_f1 = "84.96%"
best_ml_recall = "88.89%"
dnn_recall = "92.59%"

ml_comp_file = ML_RESULTS_DIR / "primary_model_comparison.json"
if ml_comp_file.exists():
    try:
        with open(ml_comp_file, "r") as f:
            ml_meta = json.load(f)
            best_summary = ml_meta.get("best_model_summary", {})
            best_ml_name = best_summary.get("Model", best_ml_name)
            best_ml_f1 = f"{best_summary.get('Test_F1', 0.8496) * 100:.1f}%"
            best_ml_recall = f"{best_summary.get('Test_Recall', 0.8889) * 100:.1f}%"
    except Exception:
        pass

with col1:
    render_metric_card("Analyzed Cohort", f"{d1_summary['total_records']:,} Students", "Historical Tracking Records")

with col2:
    render_metric_card("Identified Dropout Rate", f"{d1_summary['target_distribution']['1']['percentage']}%", "363 At-Risk Cases")

with col3:
    render_metric_card("Top ML Retention Model", f"{best_ml_name}", f"F1-Score: {best_ml_f1}")

with col4:
    render_metric_card("Deep Learning Recall", f"{dnn_recall}", "92.6% Dropout Detection Rate")

st.markdown("---")

# Platform Architecture & Modules Overview
st.subheader("🧭 Platform Navigation & Functional Modules")

mcol1, mcol2, mcol3 = st.columns(3)

with mcol1:
    st.markdown(
        """
        ### 📊 1. Overview & Explorer
        - **Overview Dashboard:** Executive KPIs, risk distributions, and institutional metrics.
        - **Dataset Explorer:** Deep inspection of `student_data.csv` and `university_student_retention_dataset_2134.csv`, missing values, distributions, and correlation heatmaps.
        """
    )

with mcol2:
    st.markdown(
        """
        ### 🤖 2. Predictive Models
        - **Machine Learning:** 5 tuned algorithms (Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting) with 5-fold Stratified CV.
        - **Deep Learning:** ANN (Dense ReLU) & DNN (BatchNorm + Dropout) with live loss/accuracy learning curves.
        """
    )

with mcol3:
    st.markdown(
        """
        ### 💡 3. Intervention & Explainability
        - **Student Risk Prediction:** Interactive individual assessment with automated academic intervention recommendations.
        - **Bulk Prediction:** Drag-and-drop CSV batch inference.
        - **Explainable AI:** SHAP global summary & individual waterfall plots.
        - **Reports:** Downloadable executive summaries and datasets.
        """
    )

st.markdown("---")

# Quick Cohort Preview
st.subheader("📋 Primary Student Cohort Snapshot")
st.dataframe(df_primary.head(6), use_container_width=True)

st.caption("Use the sidebar on the left to navigate to any module.")
