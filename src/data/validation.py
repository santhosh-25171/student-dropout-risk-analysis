"""
Schema validation, dataset compatibility checks, and target leakage detection.
"""

from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np

from src.config import (
    TARGET_COLUMN,
    PRIMARY_EXCLUDE_COLS,
    SECONDARY_EXCLUDE_COLS
)


def compare_dataset_schemas(df1: pd.DataFrame, df2: pd.DataFrame, 
                            name1: str = "Dataset 1", name2: str = "Dataset 2") -> Dict[str, Any]:
    """
    Compare schemas of two datasets to evaluate compatibility.
    Determines if direct merging is safe or if separate pipelines are required.
    """
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    common_cols = sorted(list(cols1.intersection(cols2)))
    only_in_1 = sorted(list(cols1 - cols2))
    only_in_2 = sorted(list(cols2 - cols1))

    target_compatible = False
    target_info = {}
    if TARGET_COLUMN in df1.columns and TARGET_COLUMN in df2.columns:
        t1_vals = set(df1[TARGET_COLUMN].dropna().unique())
        t2_vals = set(df2[TARGET_COLUMN].dropna().unique())
        target_info = {
            "dataset_1_unique_target_values": list(t1_vals),
            "dataset_2_unique_target_values": list(t2_vals),
            "equivalent_binary": len(t1_vals) == 2 and len(t2_vals) == 2
        }
        target_compatible = target_info["equivalent_binary"]

    # Semantic column overlap ratio
    overlap_ratio = len(common_cols) / max(len(cols1.union(cols2)), 1)
    
    recommendation = (
        "Compatible for merged modeling." if overlap_ratio > 0.8 and target_compatible
        else "Maintain separate data & modeling pipelines: Schemas differ significantly in features and semantic coverage."
    )

    return {
        "dataset_1_name": name1,
        "dataset_2_name": name2,
        "dataset_1_columns": sorted(list(cols1)),
        "dataset_2_columns": sorted(list(cols2)),
        "common_columns": common_cols,
        "only_in_dataset_1": only_in_1,
        "only_in_dataset_2": only_in_2,
        "overlap_percentage": round(overlap_ratio * 100, 2),
        "target_compatibility": target_info,
        "recommendation": recommendation,
        "safe_to_merge": False  # Enforcing separate pipelines as schemas diverge
    }


def detect_leakage_and_id_columns(df: pd.DataFrame, dataset_type: str = "primary") -> Tuple[List[str], List[str]]:
    """
    Detect identifier columns and potential target leakage features.
    """
    id_cols = [c for c in df.columns if any(k in c.lower() for k in ["student_id", "id_number", "id", "roll_no"])]
    
    # Potential leakage features (post-dropout consequences or proxies)
    leakage_keywords = ["dropout_risk", "dropout_type", "dropout_duration", "dropout_cause", "graduation_status"]
    leakage_cols = [c for c in df.columns if any(k in c.lower() for k in leakage_keywords) and c != TARGET_COLUMN]

    if dataset_type == "primary":
        id_cols = list(set(id_cols + [c for c in PRIMARY_EXCLUDE_COLS if c in df.columns]))
    else:
        id_cols = list(set(id_cols + ["student_id"] if "student_id" in df.columns else id_cols))
        leakage_cols = list(set(leakage_cols + [c for c in SECONDARY_EXCLUDE_COLS if c in df.columns and c not in id_cols]))

    return id_cols, leakage_cols


def validate_inference_data(df: pd.DataFrame, expected_features: List[str]) -> Tuple[bool, List[str], pd.DataFrame]:
    """
    Validate uploaded dataset for inference against expected features.
    Returns: (is_valid, error_messages, cleaned_df)
    """
    errors = []
    missing_cols = [col for col in expected_features if col not in df.columns]
    
    if missing_cols:
        errors.append(f"Missing required feature columns: {missing_cols}")
        return False, errors, df

    # Make a copy and select features
    clean_df = df[expected_features].copy()

    # Check for empty dataframe
    if clean_df.empty:
        errors.append("Uploaded file contains zero records.")
        return False, errors, clean_df

    return True, [], clean_df


def analyze_subgroups(df: pd.DataFrame, group_col: str, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """Analyze target distribution across demographic subgroups for fairness monitoring."""
    if group_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame()
    
    grouped = df.groupby(group_col)[target_col].agg(
        Total_Count='count',
        Dropout_Count=lambda x: (x == 1).sum(),
        Dropout_Rate=lambda x: round((x == 1).mean() * 100, 2)
    ).reset_index()
    return grouped
