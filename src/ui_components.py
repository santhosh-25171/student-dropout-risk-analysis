"""
Reusable UI layout, card styling, and navigation helper for EDUVISION AI.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from src.config import BASE_DIR, APP_TITLE, APP_SUBTITLE


def apply_custom_theme():
    """Inject custom SaaS academic CSS into Streamlit."""
    css_path = BASE_DIR / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header(title: str = APP_TITLE, subtitle: str = APP_SUBTITLE):
    """Render modern top banner."""
    st.markdown(
        f"""
        <div class="main-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric_card(title: str, value: str, subtext: str = ""):
    """Render a clean SaaS metric card."""
    st.markdown(
        f"""
        <div class="ed-metric-card">
            <div class="ed-metric-title">{title}</div>
            <div class="ed-metric-value">{value}</div>
            <div class="ed-metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_risk_badge(risk_level: str) -> str:
    """Returns HTML for color-coded risk badge."""
    if "Low" in risk_level:
        return f'<span class="badge-low">🟢 {risk_level}</span>'
    elif "Med" in risk_level:
        return f'<span class="badge-med">🟡 {risk_level}</span>'
    else:
        return f'<span class="badge-high">🔴 {risk_level}</span>'


def render_ethical_notice():
    """Render academic decision-support and ethics notice."""
    st.markdown(
        """
        <div class="callout-info">
            <strong>🛡️ Academic Decision Support Protocol:</strong>
            This system provides probabilistic risk estimates to assist educators and advisors with early intervention.
            Predictions must <em>never</em> be used as autonomous grounds for disciplinary action, grade alteration, or adverse admissions decisions.
            All risk flags require comprehensive holistic human review by trained student affairs professionals.
        </div>
        """,
        unsafe_allow_html=True
    )
