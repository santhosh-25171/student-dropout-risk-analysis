"""
Page 6: Bulk Cohort Prediction & Batch Scoring.
Allows CSV upload, schema validation, batch risk assessment, and export.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

from src.config import PRIMARY_DATASET_NAME, DEFAULT_RISK_THRESHOLDS
from src.ml.predict import predict_batch_students
from src.data.validation import validate_inference_data
from src.data.preprocessing import load_preprocessor_bundle
from src.ui_components import apply_custom_theme, render_header, render_ethical_notice

st.set_page_config(page_title="Bulk Prediction — EDUVISION AI", page_icon="📁", layout="wide")
apply_custom_theme()

render_header(
    title="📁 Bulk Cohort Risk Screening & Batch Scoring",
    subtitle="Upload new student batch datasets for institutional early-warning analysis and intervention sorting"
)

render_ethical_notice()

# Expected features
bundle = load_preprocessor_bundle("primary")
req_features = bundle["numerical_cols"] + bundle["categorical_cols"]

# Download Template Section
st.subheader("1. Sample CSV Template")
st.markdown("Download a correctly structured template CSV to populate student data for batch processing:")

from src.data.loader import load_dataset
template_df = load_dataset(PRIMARY_DATASET_NAME).drop(columns=["next_semester_dropout"]).head(5)
csv_buffer = io.StringIO()
template_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="📥 Download CSV Batch Template (5 Sample Students)",
    data=csv_buffer.getvalue(),
    file_name="eduvision_student_batch_template.csv",
    mime="text/csv"
)

st.markdown("---")

# File Upload
st.subheader("2. Upload Cohort File")
uploaded_file = st.file_uploader("Upload Student Dataset (.csv or .xlsx):", type=["csv", "xlsx", "xls"])

col_settings1, col_settings2 = st.columns(2)
with col_settings1:
    model_choice = st.selectbox(
        "Select Model for Batch Scoring:",
        ["Logistic_Regression", "Random_Forest", "Gradient_Boosting", "Support_Vector_Machine", "Decision_Tree"],
        format_func=lambda x: x.replace("_", " ")
    )
with col_settings2:
    filter_tier = st.selectbox("Filter Result Table by Risk Tier:", ["All Students", "High Risk Only", "Moderate & High Risk"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)

        st.success(f"File '{uploaded_file.name}' loaded successfully ({len(df_upload):,} records).")

        # Validation
        is_valid, errors, _ = validate_inference_data(df_upload, req_features)

        if not is_valid:
            st.error(f"Schema Validation Error: {errors}")
            st.info(f"Required features: {req_features}")
        else:
            with st.spinner("Processing batch predictions across cohort..."):
                scored_df = predict_batch_students(
                    df_upload,
                    model_name=model_choice,
                    dataset_prefix="primary",
                    thresholds=DEFAULT_RISK_THRESHOLDS
                )

            st.markdown("---")
            st.subheader("3. Cohort Screening Results")

            # Risk Summary Cards
            high_count = (scored_df["risk_level"] == "High Risk").sum()
            med_count = (scored_df["risk_level"] == "Medium Risk").sum()
            low_count = (scored_df["risk_level"] == "Low Risk").sum()

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Evaluated", f"{len(scored_df):,}")
            s2.metric("High Risk Tier", f"{high_count:,}", f"{(high_count/len(scored_df))*100:.1f}% of cohort")
            s3.metric("Medium Risk Tier", f"{med_count:,}", f"{(med_count/len(scored_df))*100:.1f}% of cohort")
            s4.metric("Low Risk Tier", f"{low_count:,}", f"{(low_count/len(scored_df))*100:.1f}% of cohort")

            # Risk Distribution Chart
            fig_risk = px.histogram(
                scored_df, x="dropout_probability", nbins=25,
                color="risk_level",
                color_discrete_map={"Low Risk": "#22c55e", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"},
                title="Cohort Risk Probability Distribution",
                labels={"dropout_probability": "Predicted Dropout Probability", "count": "Student Count"}
            )
            fig_risk.update_layout(height=320, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_risk, use_container_width=True)

            # Filter table view
            view_df = scored_df.copy()
            if filter_tier == "High Risk Only":
                view_df = view_df[view_df["risk_level"] == "High Risk"]
            elif filter_tier == "Moderate & High Risk":
                view_df = view_df[view_df["risk_level"].isin(["Medium Risk", "High Risk"])]

            # Sort descending by risk
            view_df = view_df.sort_values(by="dropout_probability", ascending=False)

            st.dataframe(
                view_df.style.background_gradient(subset=["dropout_probability"], cmap="Reds"),
                use_container_width=True
            )

            # Export Button
            out_csv = scored_df.to_csv(index=False)
            st.download_button(
                label="📥 Export Complete Scored Cohort Dataset (CSV)",
                data=out_csv,
                file_name=f"eduvision_scored_{uploaded_file.name.split('.')[0]}.csv",
                mime="text/csv",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error reading or processing uploaded file: {e}")
