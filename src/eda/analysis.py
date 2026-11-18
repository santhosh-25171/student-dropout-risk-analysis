"""
EDA Statistical Analysis and Summary Functions.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.config import TARGET_COLUMN


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute enhanced descriptive statistics for numerical columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()
    
    desc = num_df.describe().T
    desc["skewness"] = num_df.skew()
    desc["kurtosis"] = num_df.kurt()
    desc["missing_count"] = num_df.isnull().sum()
    desc["missing_pct"] = (num_df.isnull().sum() / len(num_df)) * 100
    return desc.round(3)


def compute_group_dropout_rates(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Calculate dropout rates and counts for categorical features."""
    if group_col not in df.columns or TARGET_COLUMN not in df.columns:
        return pd.DataFrame()
    
    agg = df.groupby(group_col)[TARGET_COLUMN].agg(
        Total='count',
        Dropouts=lambda x: (x == 1).sum(),
        Retained=lambda x: (x == 0).sum(),
        Dropout_Rate_Pct=lambda x: round((x == 1).mean() * 100, 2)
    ).reset_index()
    return agg.sort_values(by="Total", ascending=False)


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix of numerical features with target."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COLUMN in df.columns and TARGET_COLUMN not in num_cols:
        # Cast target to numeric if needed
        temp_df = df[num_cols].copy()
        temp_df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
        return temp_df.corr().round(3)
    return df[num_cols].corr().round(3)


def get_risk_factor_insights(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Derive automated analytical insights based on target correlations and statistics."""
    insights = []
    if TARGET_COLUMN not in df.columns:
        return insights

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET_COLUMN and "id" not in c.lower()]
    corr = df[num_cols + [TARGET_COLUMN]].corr()[TARGET_COLUMN].drop(TARGET_COLUMN)
    
    # Strongest negative correlation (protective factors)
    protective = corr.sort_values().head(2)
    for col, val in protective.items():
        if val < -0.15:
            mean_drop = df[df[TARGET_COLUMN] == 1][col].mean()
            mean_ret = df[df[TARGET_COLUMN] == 0][col].mean()
            insights.append({
                "type": "protective",
                "feature": col,
                "correlation": round(val, 3),
                "insight": f"Higher '{col}' is strongly associated with student retention (corr: {round(val, 2)}). "
                           f"Retained students averaged {round(mean_ret, 2)} vs {round(mean_drop, 2)} for dropouts."
            })

    # Strongest positive correlation (risk factors)
    risk_factors = corr.sort_values(ascending=False).head(2)
    for col, val in risk_factors.items():
        if val > 0.15:
            insights.append({
                "type": "risk",
                "feature": col,
                "correlation": round(val, 3),
                "insight": f"'{col}' shows significant positive correlation with dropout risk (corr: {round(val, 2)})."
            })

    return insights
