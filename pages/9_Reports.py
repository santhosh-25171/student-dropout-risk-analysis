"""
Page 9: Reports and Data Export Center.
Allows downloading comprehensive executive reports, model benchmarks, and scored datasets.
"""

import streamlit as st
import pandas as pd
import json
import io
import zipfile
from pathlib import Path

from src.config import (
    REPORTS_DIR,
    ML_RESULTS_DIR,
    DL_RESULTS_DIR,
    EDA_RESULTS_DIR,
    PRIMARY_DATASET_NAME
)
from src.data.loader import load_dataset, get_dataset_summary
from src.reports.export import generate_executive_markdown_report, save_project_report
from src.ui_components import apply_custom_theme, render_header

st.set_page_config(page_title="Reports & Export — EDUVISION AI", page_icon="📑", layout="wide")
apply_custom_theme()

render_header(
    title="📑 Institutional Reports & Analytical Data Exports",
    subtitle="Generate, preview, and download formal project analysis briefs, benchmarks, and scored cohorts"
)

# Load data summaries for report generation
df_primary = load_dataset(PRIMARY_DATASET_NAME)
d_summary = get_dataset_summary(df_primary)

# Load metrics
ml_data = {}
ml_file = ML_RESULTS_DIR / "primary_model_comparison.json"
if ml_file.exists():
    with open(ml_file, "r") as f:
        ml_data = json.load(f)

dl_data = {}
dl_file = DL_RESULTS_DIR / "primary_dl_evaluation.json"
if dl_file.exists():
    with open(dl_file, "r") as f:
        dl_data = json.load(f)

# Generate executive report
exec_report_md = generate_executive_markdown_report(ml_data, dl_data, d_summary)
saved_report_path = save_project_report(exec_report_md)

tab1, tab2, tab3 = st.tabs(["📄 Executive Project Brief", "📊 Model Benchmark Exports", "🖼️ Visual Asset Package"])

with tab1:
    st.subheader("Formal Academic Executive Briefing")
    
    st.markdown(exec_report_md)

    st.markdown("---")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.download_button(
            label="📥 Download Executive Brief (Markdown)",
            data=exec_report_md,
            file_name="eduvision_executive_report.md",
            mime="text/markdown",
            use_container_width=True
        )
    with dcol2:
        html_file = REPORTS_DIR / "eduvision_ai_executive_report.html"
        if html_file.exists():
            with open(html_file, "r", encoding="utf-8") as f:
                html_text = f.read()
            st.download_button(
                label="📥 Download Executive Brief (HTML)",
                data=html_text,
                file_name="eduvision_executive_report.html",
                mime="text/html",
                use_container_width=True
            )

with tab2:
    st.subheader("Quantitative Evaluation Summaries")
    
    bench_csv = DL_RESULTS_DIR / "primary_ml_vs_dl_comparison.csv"
    if bench_csv.exists():
        df_bench = pd.read_csv(bench_csv)
        st.markdown("#### Holistic Benchmark Matrix (CSV)")
        st.dataframe(df_bench, use_container_width=True)

        st.download_button(
            label="📥 Download Model Benchmark CSV",
            data=df_bench.to_csv(index=False),
            file_name="eduvision_model_benchmarks.csv",
            mime="text/csv"
        )

with tab3:
    st.subheader("Analytical Graphics Package")
    st.markdown("Package containing all generated EDA, ML ROC/PR curves, and DL training trajectory charts:")

    # Create in-memory zip of images
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for p in EDA_RESULTS_DIR.glob("*.png"):
            zip_file.write(p, arcname=f"eda/{p.name}")
        for p in ML_RESULTS_DIR.glob("*.png"):
            zip_file.write(p, arcname=f"ml/{p.name}")
        for p in DL_RESULTS_DIR.glob("*.png"):
            zip_file.write(p, arcname=f"dl/{p.name}")

    st.download_button(
        label="📦 Download Complete Visual Analytics Bundle (.ZIP)",
        data=zip_buffer.getvalue(),
        file_name="eduvision_visual_assets.zip",
        mime="application/zip",
        use_container_width=True
    )
