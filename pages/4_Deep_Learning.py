"""
Page 4: Deep Learning Architectures and Training Dynamics.
Presents ANN vs DNN architectures, loss/accuracy curves, and cross-paradigm comparisons.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path

from src.config import DL_RESULTS_DIR
from src.ui_components import apply_custom_theme, render_header, render_metric_card

st.set_page_config(page_title="Deep Learning — EDUVISION AI", page_icon="🧠", layout="wide")
apply_custom_theme()

render_header(
    title="🧠 Deep Learning Architectures & Training Dynamics",
    subtitle="Evaluation of Artificial Neural Networks (ANN) & Deep Neural Networks (DNN with BatchNorm & Dropout)"
)

# Load DL comparison and evaluation
eval_file = DL_RESULTS_DIR / "primary_dl_evaluation.json"
comp_file = DL_RESULTS_DIR / "primary_dl_comparison.csv"

if not eval_file.exists():
    st.warning("Deep Learning evaluation data not found. Please ensure training has been executed.")
    st.stop()

with open(eval_file, "r") as f:
    dl_data = json.load(f)

df_dl_comp = pd.read_csv(comp_file)

# Metric cards
ann_m = dl_data["test_metrics"]["ANN"]
dnn_m = dl_data["test_metrics"]["DNN"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("ANN Test Accuracy", f"{ann_m['accuracy'] * 100:.1f}%", "2 Dense Hidden Layers")
with c2:
    render_metric_card("ANN Test Recall", f"{ann_m['recall'] * 100:.1f}%", "Dropout Class Catch Rate")
with c3:
    render_metric_card("DNN Test Accuracy", f"{dnn_m['accuracy'] * 100:.1f}%", "4 Layers + BatchNorm + Dropout")
with c4:
    render_metric_card("DNN Test Recall", f"{dnn_m['recall'] * 100:.1f}%", "Highest Sensitivity (92.6%)")

st.markdown("---")

# Architectural Breakdown
st.subheader("🏗️ Neural Network Architecture Schematics")
acol1, acol2 = st.columns(2)

with acol1:
    st.markdown(
        """
        ### 🔹 Artificial Neural Network (ANN)
        - **Input Layer:** 14 Preprocessed Normalized Features
        - **Hidden Layer 1:** 64 Units, ReLU Activation
        - **Hidden Layer 2:** 32 Units, ReLU Activation
        - **Output Layer:** 1 Unit, Sigmoid Activation (Dropout Probability)
        - **Optimization:** Adam Optimizer ($lr=0.001$), Binary Crossentropy Loss
        - **Target Balancing:** Scikit-Learn Class Weight Balancing
        """
    )

with acol2:
    st.markdown(
        """
        ### 🔹 Deep Neural Network (DNN)
        - **Input Layer:** 14 Preprocessed Normalized Features
        - **Dense Block 1:** 128 Units $\\to$ Batch Normalization $\\to$ ReLU $\\to$ Dropout(0.30)
        - **Dense Block 2:** 64 Units $\\to$ Batch Normalization $\\to$ ReLU $\\to$ Dropout(0.30)
        - **Dense Block 3:** 32 Units $\\to$ Batch Normalization $\\to$ ReLU $\\to$ Dropout(0.20)
        - **Dense Block 4:** 16 Units $\\to$ ReLU
        - **Output Layer:** 1 Unit, Sigmoid Activation
        - **Regularization & Schedulers:** Early Stopping (patience=15) + ReduceLROnPlateau
        """
    )

st.markdown("---")

# Learning Curves (Loss & Accuracy)
st.subheader("📈 Training & Validation Epoch Trajectories")
lcol1, lcol2 = st.columns(2)

loss_img = DL_RESULTS_DIR / "primary_loss_curves.png"
acc_img = DL_RESULTS_DIR / "primary_accuracy_curves.png"

with lcol1:
    if loss_img.exists():
        st.image(str(loss_img), caption="Training vs Validation Loss (Binary Crossentropy)", use_container_width=True)

with lcol2:
    if acc_img.exists():
        st.image(str(acc_img), caption="Training vs Validation Accuracy", use_container_width=True)

st.markdown("---")

# Confusion Matrices
st.subheader("🎯 Deep Learning Confusion Matrices (Holdout Test Set)")
cm_col1, cm_col2 = st.columns(2)

with cm_col1:
    ann_cm = DL_RESULTS_DIR / "primary_ANN_cm.png"
    if ann_cm.exists():
        st.image(str(ann_cm), caption="ANN Confusion Matrix", use_container_width=True)

with cm_col2:
    dnn_cm = DL_RESULTS_DIR / "primary_DNN_cm.png"
    if dnn_cm.exists():
        st.image(str(dnn_cm), caption="DNN Confusion Matrix", use_container_width=True)

st.markdown("---")

# ANN vs DNN Comparison Table
st.subheader("📊 ANN vs DNN Test Performance Matrix")
st.dataframe(df_dl_comp, use_container_width=True)
