"""
Page 3: Machine Learning Model Benchmarking & Analytics.
Displays performance metrics, confusion matrices, ROC/PR curves, and feature importance across 5 algorithms.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

from src.config import ML_RESULTS_DIR
from src.ui_components import apply_custom_theme, render_header, render_metric_card

st.set_page_config(page_title="Machine Learning — EDUVISION AI", page_icon="🤖", layout="wide")
apply_custom_theme()

render_header(
    title="🤖 Machine Learning Benchmark & Diagnostics",
    subtitle="Comparative evaluation of 5 tuned algorithms with 5-fold Stratified Cross-Validation"
)

# Load comparison table
comp_file = ML_RESULTS_DIR / "primary_model_comparison.csv"
meta_file = ML_RESULTS_DIR / "primary_model_comparison.json"

if not comp_file.exists():
    st.warning("ML results not found. Please ensure model training has been executed.")
    st.stop()

df_comp = pd.read_csv(comp_file)
with open(meta_file, "r") as f:
    meta_data = json.load(f)

# Top metrics cards
top_model = df_comp.iloc[0]
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_metric_card("Selected Champion Model", top_model["Model"], "Optimized for Recall & F1")
with k2:
    render_metric_card("Test Accuracy", f"{top_model['Test_Accuracy'] * 100:.1f}%", "Untouched 15% Holdout")
with k3:
    render_metric_card("Test Recall (Sensitivity)", f"{top_model['Test_Recall'] * 100:.1f}%", "Dropout Catch Rate")
with k4:
    render_metric_card("Test ROC-AUC", f"{top_model['Test_ROC_AUC']:.4f}", "Discriminative Power")

st.markdown("---")

# Main comparison table
st.subheader("📊 Complete 5-Algorithm Performance Matrix (Test Partition)")
st.dataframe(
    df_comp.style.highlight_max(subset=["Test_Accuracy", "Test_Recall", "Test_F1", "Test_ROC_AUC"], color="#dcfce7"),
    use_container_width=True
)

st.markdown("---")

# Diagnostic Curves
st.subheader("📈 Diagnostic Curves Comparison")
ccol1, ccol2 = st.columns(2)

roc_img = ML_RESULTS_DIR / "primary_roc_curves.png"
pr_img = ML_RESULTS_DIR / "primary_pr_curves.png"

with ccol1:
    if roc_img.exists():
        st.image(str(roc_img), caption="Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)

with ccol2:
    if pr_img.exists():
        st.image(str(pr_img), caption="Precision-Recall Curve Comparison", use_container_width=True)

st.markdown("---")

# Detailed Model Inspector
st.subheader("🔍 Individual Algorithm Inspector")
model_names = [m.replace(" ", "_") for m in df_comp["Model"].tolist()]
selected_model = st.selectbox("Select Algorithm to Inspect in Detail:", model_names, format_func=lambda x: x.replace("_", " "))

dcol1, dcol2 = st.columns(2)

with dcol1:
    cm_path = ML_RESULTS_DIR / f"primary_{selected_model}_cm.png"
    if cm_path.exists():
        st.image(str(cm_path), caption=f"Confusion Matrix ({selected_model.replace('_', ' ')})", use_container_width=True)

with dcol2:
    fi_path = ML_RESULTS_DIR / f"primary_{selected_model}_feat_imp.png"
    if fi_path.exists():
        st.image(str(fi_path), caption=f"Feature Importances / Coefficients ({selected_model.replace('_', ' ')})", use_container_width=True)
    else:
        st.info("Feature importance not directly available for this model kernel.")

# Classification Report
st.markdown("#### Detailed Classification Report")
test_metrics = meta_data.get("test_metrics", {}).get(selected_model, {})
rep = test_metrics.get("classification_report", {})
if rep:
    rep_df = pd.DataFrame(rep).T.round(3)
    st.dataframe(rep_df, use_container_width=True)
