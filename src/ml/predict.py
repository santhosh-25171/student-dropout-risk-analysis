"""
Inference engine for individual and bulk student dropout risk predictions.
"""

from typing import Dict, Any, List, Union, Tuple, Optional
import pandas as pd
import numpy as np
import joblib

from src.config import (
    DEFAULT_RISK_THRESHOLDS,
    ML_MODELS_DIR,
    PREPROCESSING_DIR
)
from src.data.preprocessing import load_preprocessor_bundle


def assign_risk_level(prob: float, thresholds: Optional[Dict[str, float]] = None) -> Tuple[str, str]:
    """
    Map dropout probability to standard risk classification and badge color.
    """
    t = thresholds or DEFAULT_RISK_THRESHOLDS
    low_max = t.get("low_max", 0.30)
    med_max = t.get("medium_max", 0.65)

    if prob < low_max:
        return "Low Risk", "#27ae60"  # Green
    elif prob < med_max:
        return "Medium Risk", "#f39c12"  # Amber
    else:
        return "High Risk", "#e74c3c"  # Red


def get_recommended_interventions(student_data: Dict[str, Any], prob: float) -> List[str]:
    """Generate prioritized academic support recommendations based on student risk profile."""
    actions = []
    
    # Check GPA
    gpa = student_data.get("current_gpa", student_data.get("gpa_current", 3.0))
    if float(gpa) < 2.5:
        actions.append("Assign peer academic tutoring and schedule bi-weekly GPA check-ins.")
    
    # Check Attendance
    att = student_data.get("attendance_rate", 85.0)
    if float(att) < 75.0:
        actions.append("Initiate proactive attendance counseling and identify commute or scheduling barriers.")

    # Check Financial Aid
    aid = student_data.get("financial_aid", student_data.get("financial_aid_status", 1))
    if aid == 0 or str(aid).lower() in ["false", "0", "no"]:
        actions.append("Connect student with emergency bursary and institutional scholarship advisor.")

    # Check Part-time Job
    job = student_data.get("part_time_job", 0)
    if job == 1 or str(job).lower() in ["true", "1", "yes"]:
        actions.append("Provide time-management coaching and flexible course scheduling options.")

    if not actions:
        if prob >= 0.5:
            actions.append("Conduct a 1-on-1 advisor review to assess holistic well-being and academic workload.")
        else:
            actions.append("Maintain standard academic advisory schedule; student is on positive retention trajectory.")

    return actions


def predict_single_student(
    student_dict: Dict[str, Any],
    model_name: str = "Random_Forest",
    dataset_prefix: str = "primary",
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Generate dropout prediction for an individual student record.
    """
    # Load preprocessing bundle
    bundle = load_preprocessor_bundle(dataset_prefix)
    preprocessor = bundle["preprocessor"]

    # Load model
    model_path = ML_MODELS_DIR / f"{dataset_prefix}_{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
    model = joblib.load(model_path)

    # Convert to DataFrame
    df_input = pd.DataFrame([student_dict])
    
    # Preprocess
    X_proc = preprocessor.transform(df_input)

    # Inference
    pred_class = int(model.predict(X_proc)[0])
    prob = float(model.predict_proba(X_proc)[0, 1]) if hasattr(model, "predict_proba") else float(pred_class)
    
    risk_level, color = assign_risk_level(prob, thresholds)
    recommendations = get_recommended_interventions(student_dict, prob)

    return {
        "model_used": model_name.replace("_", " "),
        "prediction_class": pred_class,
        "prediction_label": "Dropout Risk" if pred_class == 1 else "Likely Retained",
        "dropout_probability": round(prob, 4),
        "dropout_probability_pct": round(prob * 100, 1),
        "risk_level": risk_level,
        "risk_color": color,
        "recommendations": recommendations,
        "raw_input": student_dict,
        "processed_features": X_proc
    }


def predict_batch_students(
    df_raw: pd.DataFrame,
    model_name: str = "Random_Forest",
    dataset_prefix: str = "primary",
    thresholds: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Generate batch predictions on an uploaded dataset.
    Preserves all input columns and appends risk assessment metrics.
    """
    bundle = load_preprocessor_bundle(dataset_prefix)
    preprocessor = bundle["preprocessor"]

    model_path = ML_MODELS_DIR / f"{dataset_prefix}_{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_name} not found.")
    model = joblib.load(model_path)

    # Extract required feature columns
    req_num = bundle["numerical_cols"]
    req_cat = bundle["categorical_cols"]
    all_features = req_num + req_cat

    # Ensure required features exist or fill defaults
    df_features = df_raw.copy()
    for col in all_features:
        if col not in df_features.columns:
            df_features[col] = np.nan

    X_proc = preprocessor.transform(df_features[all_features])

    preds = model.predict(X_proc)
    probas = model.predict_proba(X_proc)[:, 1] if hasattr(model, "predict_proba") else preds.astype(float)

    results_df = df_raw.copy()
    results_df["predicted_dropout"] = preds
    results_df["predicted_status"] = np.where(preds == 1, "Dropout Risk", "Retained")
    results_df["dropout_probability"] = np.round(probas, 4)
    results_df["dropout_probability_pct"] = np.round(probas * 100, 1)

    t = thresholds or DEFAULT_RISK_THRESHOLDS
    results_df["risk_level"] = [assign_risk_level(p, t)[0] for p in probas]

    return results_df
