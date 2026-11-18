"""
Unit tests for data loading, summary calculation, and validation modules.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.loader import load_dataset, get_dataset_summary
from src.data.validation import compare_dataset_schemas, detect_leakage_and_id_columns, validate_inference_data
from src.config import TARGET_COLUMN


def test_load_primary_dataset():
    df = load_dataset("student_data.csv")
    assert not df.empty
    assert len(df) == 1000
    assert TARGET_COLUMN in df.columns
    # Check target standardized to binary integer
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_load_secondary_dataset():
    df = load_dataset("university_student_retention_dataset_2134.csv")
    assert not df.empty
    assert len(df) == 2134
    assert TARGET_COLUMN in df.columns
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_get_dataset_summary():
    df = load_dataset("student_data.csv")
    summary = get_dataset_summary(df)
    assert summary["total_records"] == 1000
    assert summary["total_columns"] == 12
    assert summary["duplicate_rows"] == 0
    assert "target_distribution" in summary
    assert "0" in summary["target_distribution"]
    assert "1" in summary["target_distribution"]


def test_compare_dataset_schemas():
    df1 = load_dataset("student_data.csv")
    df2 = load_dataset("university_student_retention_dataset_2134.csv")
    comparison = compare_dataset_schemas(df1, df2)
    assert "common_columns" in comparison
    assert "overlap_percentage" in comparison
    assert comparison["safe_to_merge"] is False  # Enforced separate pipelines


def test_detect_leakage_columns():
    df2 = load_dataset("university_student_retention_dataset_2134.csv")
    id_cols, leak_cols = detect_leakage_and_id_columns(df2, dataset_type="secondary")
    assert "student_id" in id_cols
    assert "dropout_risk" in leak_cols
    assert "dropout_type" in leak_cols


def test_validate_inference_data():
    df = load_dataset("student_data.csv")
    expected = ["age", "attendance_rate", "current_gpa"]
    is_valid, errors, clean = validate_inference_data(df, expected)
    assert is_valid is True
    assert len(errors) == 0
    assert list(clean.columns) == expected
