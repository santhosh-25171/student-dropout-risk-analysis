"""
Unit tests for student dropout inference, model loading, and thresholding.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.loader import load_dataset
from src.ml.predict import (
    assign_risk_level,
    predict_single_student,
    predict_batch_students,
    get_recommended_interventions
)


def test_assign_risk_level():
    lvl_low, col_low = assign_risk_level(0.15)
    assert lvl_low == "Low Risk"
    
    lvl_med, col_med = assign_risk_level(0.45)
    assert lvl_med == "Medium Risk"
    
    lvl_high, col_high = assign_risk_level(0.85)
    assert lvl_high == "High Risk"


def test_single_student_prediction():
    student = {
        "age": 20,
        "gender": "Female",
        "socioeconomic_status": "Medium",
        "commute_distance": 12.0,
        "attendance_rate": 88.0,
        "previous_grades": 82.0,
        "current_gpa": 3.4,
        "financial_aid": 1,
        "part_time_job": 0,
        "extracurricular_activities": 1
    }
    
    res = predict_single_student(student, model_name="Logistic_Regression", dataset_prefix="primary")
    assert "prediction_class" in res
    assert res["prediction_class"] in [0, 1]
    assert 0.0 <= res["dropout_probability"] <= 1.0
    assert res["risk_level"] in ["Low Risk", "Medium Risk", "High Risk"]
    assert len(res["recommendations"]) > 0


def test_batch_prediction():
    df = load_dataset("student_data.csv").head(20)
    res_df = predict_batch_students(df, model_name="Logistic_Regression", dataset_prefix="primary")
    assert "predicted_dropout" in res_df.columns
    assert "predicted_status" in res_df.columns
    assert "dropout_probability" in res_df.columns
    assert "risk_level" in res_df.columns
    assert len(res_df) == 20
