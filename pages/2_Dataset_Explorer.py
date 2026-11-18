"""
Page 2: Dataset Explorer and Interactive Exploratory Data Analysis.
Supports dataset switching, filtering, missing-value diagnostics, and EDA visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.config import (
    PRIMARY_DATASET_NAME,
    SECONDARY_DATASET_NAME,
    EDA_RESULTS_DIR,
    TARGET_COLUMN
)
from src.data.loader import load_dataset, get_dataset_summary
from src.data.validation import compare_dataset_schemas
from src.ui_components import apply_custom_theme, render_header

st.set_page_config(page_title="Dataset Explorer — EDUVISION AI", page_icon="🔍", layout="wide")
apply_custom_theme()

render_header(
    title="🔍 Institutional Dataset Explorer & Schema Audit",
    subtitle="Comprehensive exploratory data analysis, missing-data auditing, and cohort segmentation"
)

# Dataset Selector
dcol1, dcol2 = st.columns([2, 1])
with dcol1:
    dataset_option = st.selectbox(
        "Select Dataset to Inspect:",
        [
            f"Primary Predictive Dataset ({PRIMARY_DATASET_NAME})",
            f"Institutional Multi-Semester Cohort ({SECONDARY_DATASET_NAME})"
        ]
    )

selected_file = PRIMARY_DATASET_NAME if "Primary" in dataset_option else SECONDARY_DATASET_NAME
df = load_dataset(selected_file)
summary = get_dataset_summary(df)

with dcol2:
    st.metric("Total Records", f"{summary['total_records']:,}")
    st.metric("Total Features", f"{summary['total_columns']}")

tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview & Filters", "📊 Statistical Summaries", "📉 Missing Values & Audit", "📈 Visual EDA Gallery"])

with tab1:
    st.subheader("Data Preview with Interactive Filtering")
    
    fcol1, fcol2, fcol3 = st.columns(3)
    filtered_df = df.copy()

    # Dynamic filter columns if present
    if "gender" in df.columns:
        with fcol1:
            g_opts = ["All"] + list(df["gender"].dropna().unique())
            selected_gender = st.selectbox("Filter by Gender:", g_opts)
            if selected_gender != "All":
                filtered_df = filtered_df[filtered_df["gender"] == selected_gender]

    gpa_col = "current_gpa" if "current_gpa" in df.columns else ("gpa_current" if "gpa_current" in df.columns else None)
    if gpa_col:
        with fcol2:
            min_g, max_g = float(df[gpa_col].min()), float(df[gpa_col].max())
            gpa_range = st.slider("Filter by GPA Range:", min_g, max_g, (min_g, max_g), step=0.1)
            filtered_df = filtered_df[(filtered_df[gpa_col] >= gpa_range[0]) & (filtered_df[gpa_col] <= gpa_range[1])]

    if TARGET_COLUMN in df.columns:
        with fcol3:
            t_opts = ["All", "Retained (0)", "Dropout Risk (1)"]
            selected_t = st.selectbox("Filter by Status:", t_opts)
            if selected_t == "Retained (0)":
                filtered_df = filtered_df[filtered_df[TARGET_COLUMN] == 0]
            elif selected_t == "Dropout Risk (1)":
                filtered_df = filtered_df[filtered_df[TARGET_COLUMN] == 1]

    st.write(f"Showing **{len(filtered_df):,}** of **{len(df):,}** records matching filters:")
    st.dataframe(filtered_df.head(100), use_container_width=True)

with tab2:
    st.subheader("Descriptive Statistics")
    st.markdown("#### Numerical Attributes")
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        st.dataframe(num_df.describe().T.round(2), use_container_width=True)

    st.markdown("#### Categorical Attributes")
    cat_df = df.select_dtypes(exclude=[np.number])
    if not cat_df.empty:
        st.dataframe(cat_df.describe().T, use_container_width=True)

with tab3:
    st.subheader("Missing Values & Data Integrity Audit")
    mcol1, mcol2 = st.columns([1, 1])
    
    with mcol1:
        st.markdown(f"**Total Missing Cells:** {summary['total_missing_cells']}")
        st.markdown(f"**Duplicate Rows:** {summary['duplicate_rows']}")
        
        missing_dict = summary["missing_values"]
        if missing_dict:
            miss_df = pd.DataFrame(list(missing_dict.items()), columns=["Column", "Missing Count"])
            miss_df["Missing Percentage (%)"] = (miss_df["Missing Count"] / len(df)) * 100
            st.dataframe(miss_df, use_container_width=True)
        else:
            st.success("✅ Zero missing values detected across all columns in this dataset.")

    with mcol2:
        st.markdown("#### Cross-Dataset Schema Comparison")
        df1 = load_dataset(PRIMARY_DATASET_NAME)
        df2 = load_dataset(SECONDARY_DATASET_NAME)
        comp = compare_dataset_schemas(df1, df2, PRIMARY_DATASET_NAME, SECONDARY_DATASET_NAME)
        st.markdown(f"**Common Feature Overlap:** {comp['overlap_percentage']}%")
        st.markdown(f"**Common Columns:** `{', '.join(comp['common_columns'])}`")
        st.markdown(f"**Only in Primary:** `{', '.join(comp['only_in_dataset_1'])}`")
        st.markdown(f"**Only in Secondary:** `{', '.join(comp['only_in_dataset_2'][:6])}...`")
        st.info(f"💡 {comp['recommendation']}")

with tab4:
    st.subheader("Exploratory Data Analysis Gallery")
    st.markdown("High-resolution analytical figures generated directly from the historical dataset:")

    eda_files = sorted(list(EDA_RESULTS_DIR.glob("*.png")))
    if eda_files:
        for i in range(0, len(eda_files), 2):
            cols = st.columns(2)
            with cols[0]:
                st.image(str(eda_files[i]), use_container_width=True)
                st.caption(eda_files[i].stem.replace("_", " ").title())
            if i + 1 < len(eda_files):
                with cols[1]:
                    st.image(str(eda_files[i + 1]), use_container_width=True)
                    st.caption(eda_files[i + 1].stem.replace("_", " ").title())
    else:
        st.info("EDA charts are being generated. Please wait...")
