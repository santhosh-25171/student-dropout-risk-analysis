"""
Dataset loading utilities with validation and caching support.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_DIR,
    BASE_DIR,
    PRIMARY_DATASET_NAME,
    SECONDARY_DATASET_NAME,
    TARGET_COLUMN
)


def resolve_dataset_path(dataset_name: str) -> Path:
    """Resolve dataset path checking raw directory and root workspace."""
    candidate_paths = [
        RAW_DATA_DIR / dataset_name,
        BASE_DIR / dataset_name,
        Path(dataset_name)
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Dataset '{dataset_name}' not found in raw data or workspace.")


def load_dataset(dataset_name: str = PRIMARY_DATASET_NAME) -> pd.DataFrame:
    """Load a dataset by name, standardizing target format and boolean columns."""
    path = resolve_dataset_path(dataset_name)
    if str(path).endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Standardize target variable to binary integer 0 / 1 if present
    if TARGET_COLUMN in df.columns:
        if df[TARGET_COLUMN].dtype == bool:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
        elif df[TARGET_COLUMN].dtype == object:
            # Handle string boolean representations
            mapping = {'true': 1, 'false': 0, '1': 1, '0': 0, 'yes': 1, 'no': 0, 'dropout': 1, 'retained': 0}
            df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().str.lower().map(mapping)

    return df


def get_dataset_summary(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Dict[str, Any]:
    """Calculate comprehensive dataset inspection metrics."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    target_dist = None
    target_imbalance_ratio = None
    if target_col in df.columns:
        counts = df[target_col].value_counts(dropna=False).to_dict()
        proportions = df[target_col].value_counts(normalize=True, dropna=False).to_dict()
        target_dist = {str(k): {"count": int(v), "percentage": round(float(proportions.get(k, 0)) * 100, 2)} 
                       for k, v in counts.items()}
        # Calculate imbalance ratio if binary
        if len(counts) == 2:
            sorted_counts = sorted(counts.values())
            target_imbalance_ratio = round(sorted_counts[1] / max(1, sorted_counts[0]), 2)

    return {
        "total_records": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items() if v > 0},
        "total_missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": target_dist,
        "target_imbalance_ratio": target_imbalance_ratio,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }
