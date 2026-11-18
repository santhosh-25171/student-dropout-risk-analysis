"""
Unit tests for data preprocessing and leakage prevention.
"""

import pytest
import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.data.preprocessing import (
    get_feature_target_split,
    create_train_val_test_splits,
    fit_and_transform_pipeline,
    load_preprocessor_bundle
)


def test_feature_target_split():
    df = load_dataset("student_data.csv")
    X, y, num_cols, cat_cols = get_feature_target_split(df, dataset_type="primary")
    assert "student_id" not in X.columns
    assert "next_semester_dropout" not in X.columns
    assert len(num_cols) > 0
    assert len(cat_cols) > 0
    assert len(y) == len(X)


def test_train_val_test_splits_stratification():
    df = load_dataset("student_data.csv")
    X, y, _, _ = get_feature_target_split(df, dataset_type="primary")
    splits = create_train_val_test_splits(X, y, test_size=0.15, val_size=0.15, random_state=42)

    # Total check
    total_split_len = len(splits["y_train"]) + len(splits["y_val"]) + len(splits["y_test"])
    assert total_split_len == len(y)
    assert len(splits["y_test"]) == 150
    assert len(splits["y_val"]) == 150
    assert len(splits["y_train"]) == 700

    # Stratification check (similar positive ratios)
    train_rate = splits["y_train"].mean()
    val_rate = splits["y_val"].mean()
    test_rate = splits["y_test"].mean()
    assert abs(train_rate - test_rate) < 0.05
    assert abs(val_rate - test_rate) < 0.05


def test_preprocessor_fitting_and_loading():
    df = load_dataset("student_data.csv")
    X, y, num_cols, cat_cols = get_feature_target_split(df, dataset_type="primary")
    splits = create_train_val_test_splits(X, y)
    bundle = fit_and_transform_pipeline(splits, num_cols, cat_cols, dataset_name="primary")

    assert bundle["X_train_proc"].shape[0] == 700
    assert bundle["X_test_proc"].shape[0] == 150
    assert len(bundle["feature_names"]) == bundle["X_train_proc"].shape[1]

    # Test loading
    loaded_bundle = load_preprocessor_bundle("primary")
    assert loaded_bundle["preprocessor"] is not None
    assert loaded_bundle["feature_names"] == bundle["feature_names"]
