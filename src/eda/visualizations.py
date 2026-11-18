"""
Comprehensive EDA visualization generators using Matplotlib, Seaborn, and Plotly.
Saves high-resolution publication-ready figures to results/eda.
"""

from pathlib import Path
from typing import Optional, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.config import EDA_RESULTS_DIR, TARGET_COLUMN

# Style configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def save_fig(fig, filename: str) -> Path:
    """Save matplotlib figure to results/eda/ directory."""
    EDA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EDA_RESULTS_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_dropout_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot target class distribution (retained vs dropout)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET_COLUMN].value_counts().rename(index={0: "Retained (0)", 1: "Dropout (1)"})
    palette = ["#2b5c8f", "#d9534f"]
    
    bars = ax.bar(counts.index, counts.values, color=palette, width=0.5, edgecolor="none", alpha=0.9)
    ax.set_title("Student Retention vs Dropout Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Student Count", fontsize=11)
    
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.annotate(f"{h:,}\n({pct:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                    xytext=(0, 0), textcoords="offset points", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=10)
    
    if save:
        save_fig(fig, "01_dropout_distribution.png")
    return fig


def plot_demographics(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot gender and demographic breakdown with dropout proportions."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if "gender" in df.columns:
        cross = pd.crosstab(df["gender"], df[TARGET_COLUMN], normalize="index") * 100
        cross.plot(kind="bar", stacked=True, color=["#2b5c8f", "#d9534f"], ax=ax, width=0.5)
        ax.set_title("Dropout Proportion by Student Gender", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Gender", fontsize=11)
        ax.set_ylabel("Percentage (%)", fontsize=11)
        ax.legend(["Retained", "Dropout"], loc="upper right")
        plt.xticks(rotation=0)
    if save:
        save_fig(fig, "02_demographic_distribution.png")
    return fig


def plot_age_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot age distribution by dropout status."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if "age" in df.columns:
        sns.histplot(data=df, x="age", hue=TARGET_COLUMN, multiple="stack", 
                     palette=["#2b5c8f", "#d9534f"], bins=10, ax=ax, shrink=0.85)
        ax.set_title("Age Distribution by Retention Status", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Age", fontsize=11)
        ax.set_ylabel("Student Count", fontsize=11)
    if save:
        save_fig(fig, "03_age_distribution.png")
    return fig


def plot_attendance_analysis(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot attendance rate analysis and KDE."""
    fig, ax = plt.subplots(figsize=(7, 4))
    col = "attendance_rate" if "attendance_rate" in df.columns else None
    if col:
        sns.kdeplot(data=df[df[TARGET_COLUMN] == 0], x=col, label="Retained", color="#2b5c8f", fill=True, alpha=0.3, ax=ax)
        sns.kdeplot(data=df[df[TARGET_COLUMN] == 1], x=col, label="Dropout", color="#d9534f", fill=True, alpha=0.3, ax=ax)
        ax.set_title("Attendance Rate Density by Retention Status", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Attendance Rate (%)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend()
    if save:
        save_fig(fig, "04_attendance_analysis.png")
    return fig


def plot_gpa_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot GPA violin and box distribution by dropout status."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    gpa_col = "current_gpa" if "current_gpa" in df.columns else ("gpa_current" if "gpa_current" in df.columns else None)
    if gpa_col:
        sns.boxplot(data=df, x=TARGET_COLUMN, y=gpa_col, hue=TARGET_COLUMN, palette=["#2b5c8f", "#d9534f"], ax=ax, width=0.4, legend=False)
        ax.set_title(f"Cumulative GPA Distribution ({gpa_col})", fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Retained", "Dropout"])
        ax.set_xlabel("Student Status", fontsize=11)
        ax.set_ylabel("GPA (4.0 scale)", fontsize=11)
    if save:
        save_fig(fig, "05_gpa_distribution.png")
    return fig


def plot_socioeconomic_analysis(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot socioeconomic status vs dropout rate."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if "socioeconomic_status" in df.columns:
        rates = df.groupby("socioeconomic_status")[TARGET_COLUMN].mean().reset_index()
        rates[TARGET_COLUMN] = rates[TARGET_COLUMN] * 100
        # Reorder Low, Medium, High if present
        order = [s for s in ["Low", "Medium", "High"] if s in rates["socioeconomic_status"].values]
        sns.barplot(data=rates, x="socioeconomic_status", y=TARGET_COLUMN, hue="socioeconomic_status", order=order, palette="Blues_r", ax=ax, legend=False)
        ax.set_title("Dropout Rate by Socioeconomic Status", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Socioeconomic Tier", fontsize=11)
        ax.set_ylabel("Dropout Rate (%)", fontsize=11)
    if save:
        save_fig(fig, "06_socioeconomic_analysis.png")
    return fig


def plot_financial_aid_analysis(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot financial aid status comparison."""
    fig, ax = plt.subplots(figsize=(6, 4))
    col = "financial_aid" if "financial_aid" in df.columns else ("financial_aid_status" if "financial_aid_status" in df.columns else None)
    if col:
        rates = df.groupby(col)[TARGET_COLUMN].mean() * 100
        bars = ax.bar(["No Financial Aid", "Receiving Aid"], rates.values, color=["#d9534f", "#2b5c8f"], width=0.45)
        ax.set_title("Dropout Rate by Financial Aid Support", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Dropout Rate (%)", fontsize=11)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h / 2),
                        ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    if save:
        save_fig(fig, "07_financial_aid_analysis.png")
    return fig


def plot_employment_analysis(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot part-time job impact."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if "part_time_job" in df.columns:
        rates = df.groupby("part_time_job")[TARGET_COLUMN].mean() * 100
        bars = ax.bar(["No Part-Time Job", "Working Part-Time"], rates.values, color=["#2b5c8f", "#d9534f"], width=0.45)
        ax.set_title("Dropout Rate by Part-Time Employment", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Dropout Rate (%)", fontsize=11)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h / 2),
                        ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    if save:
        save_fig(fig, "08_employment_analysis.png")
    return fig


def plot_extracurricular_analysis(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot extracurricular involvement comparison."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if "extracurricular_activities" in df.columns:
        rates = df.groupby("extracurricular_activities")[TARGET_COLUMN].mean() * 100
        bars = ax.bar(["No Activities", "Active Participant"], rates.values, color=["#d9534f", "#2b5c8f"], width=0.45)
        ax.set_title("Dropout Rate by Extracurricular Participation", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Dropout Rate (%)", fontsize=11)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h / 2),
                        ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    if save:
        save_fig(fig, "09_extracurricular_analysis.png")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot annotated correlation matrix."""
    fig, ax = plt.subplots(figsize=(9, 7))
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if "id" not in c.lower()]
    corr = df[num_cols].corr()
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap of Academic and Behavioral Features", fontsize=13, fontweight="bold", pad=12)
    if save:
        save_fig(fig, "10_correlation_heatmap.png")
    return fig


def plot_feature_relationships(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot GPA vs Attendance Rate scatter with dropout hue and marginals."""
    gpa_col = "current_gpa" if "current_gpa" in df.columns else ("gpa_current" if "gpa_current" in df.columns else None)
    att_col = "attendance_rate" if "attendance_rate" in df.columns else None
    
    fig, ax = plt.subplots(figsize=(8, 5))
    if gpa_col and att_col:
        sns.scatterplot(
            data=df, x=att_col, y=gpa_col, hue=TARGET_COLUMN,
            palette=["#2b5c8f", "#d9534f"], alpha=0.7, s=40, ax=ax
        )
        ax.set_title(f"Interplay of {att_col} and {gpa_col} on Retention", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Attendance Rate (%)", fontsize=11)
        ax.set_ylabel("GPA (4.0 Scale)", fontsize=11)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=["Retained", "Dropout Risk"])
    if save:
        save_fig(fig, "11_feature_relationships.png")
    return fig


def generate_all_eda_visualizations(df: pd.DataFrame) -> Dict[str, Path]:
    """Execute all EDA plots and return dict of saved image paths."""
    EDA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "dropout_distribution": save_fig(plot_dropout_distribution(df, save=False), "01_dropout_distribution.png"),
        "demographics": save_fig(plot_demographics(df, save=False), "02_demographic_distribution.png"),
        "age": save_fig(plot_age_distribution(df, save=False), "03_age_distribution.png"),
        "attendance": save_fig(plot_attendance_analysis(df, save=False), "04_attendance_analysis.png"),
        "gpa": save_fig(plot_gpa_distribution(df, save=False), "05_gpa_distribution.png"),
        "socioeconomic": save_fig(plot_socioeconomic_analysis(df, save=False), "06_socioeconomic_analysis.png"),
        "financial_aid": save_fig(plot_financial_aid_analysis(df, save=False), "07_financial_aid_analysis.png"),
        "employment": save_fig(plot_employment_analysis(df, save=False), "08_employment_analysis.png"),
        "extracurricular": save_fig(plot_extracurricular_analysis(df, save=False), "09_extracurricular_analysis.png"),
        "correlation": save_fig(plot_correlation_heatmap(df, save=False), "10_correlation_heatmap.png"),
        "relationships": save_fig(plot_feature_relationships(df, save=False), "11_feature_relationships.png")
    }
    return paths
