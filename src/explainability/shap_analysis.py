"""
Explainable AI (XAI) Engine powered by SHAP.
Generates global feature attributions and individual student prediction breakdowns.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import joblib

from src.config import EXPLAIN_RESULTS_DIR, ML_MODELS_DIR
from src.data.preprocessing import load_preprocessor_bundle


def get_model_explainer(model: Any, background_data: np.ndarray) -> shap.Explainer:
    """Instantiate appropriate SHAP explainer based on model architecture."""
    model_type = type(model).__name__
    
    if any(k in model_type for k in ["RandomForest", "DecisionTree", "GradientBoosting"]):
        try:
            return shap.TreeExplainer(model)
        except Exception:
            return shap.Explainer(model.predict, background_data)
    elif "LogisticRegression" in model_type:
        try:
            return shap.LinearExplainer(model, background_data)
        except Exception:
            return shap.Explainer(model.predict_proba, background_data)
    else:
        # Fallback kernel / general explainer with limited background to keep speed fast
        sample_bg = background_data[:min(50, len(background_data))]
        return shap.Explainer(model.predict, sample_bg)


def compute_global_shap(
    model_name: str = "Random_Forest",
    dataset_prefix: str = "primary",
    max_samples: int = 150
) -> Dict[str, Any]:
    """
    Compute global SHAP values across test dataset samples.
    Generates Beeswarm summary and Bar importance plots.
    """
    EXPLAIN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    bundle = load_preprocessor_bundle(dataset_prefix)
    feature_names = bundle["feature_names"]
    
    # Load model
    model_path = ML_MODELS_DIR / f"{dataset_prefix}_{model_name}.joblib"
    model = joblib.load(model_path)
    
    # Load training/test data
    from src.data.loader import load_dataset
    from src.data.preprocessing import get_feature_target_split, create_train_val_test_splits
    
    csv_name = "student_data.csv" if dataset_prefix == "primary" else "university_student_retention_dataset_2134.csv"
    df = load_dataset(csv_name)
    X, y, num_cols, cat_cols = get_feature_target_split(df, dataset_type=dataset_prefix)
    splits = create_train_val_test_splits(X, y)
    
    X_test_proc = bundle["preprocessor"].transform(splits["X_test"][:max_samples])
    X_train_proc = bundle["preprocessor"].transform(splits["X_train"][:100])
    
    explainer = get_model_explainer(model, X_train_proc)
    
    # Calculate shap values
    raw_shap = explainer(X_test_proc)
    
    # Extract values for positive class (index 1) if multi-dimensional
    if len(raw_shap.shape) == 3 and raw_shap.shape[2] == 2:
        shap_vals = raw_shap[:, :, 1]
    else:
        shap_vals = raw_shap
        
    shap_vals.feature_names = feature_names

    # 1. Beeswarm Plot
    fig_summary, ax = plt.subplots(figsize=(8, 6))
    shap.plots.beeswarm(shap_vals, max_display=10, show=False)
    plt.title(f"Global SHAP Impact: {model_name.replace('_', ' ')}", fontsize=12, fontweight="bold", pad=12)
    beeswarm_path = EXPLAIN_RESULTS_DIR / f"{dataset_prefix}_{model_name}_shap_summary.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Bar Plot
    fig_bar, ax = plt.subplots(figsize=(8, 5))
    shap.plots.bar(shap_vals, max_display=10, show=False)
    plt.title(f"Mean Absolute SHAP Feature Importance: {model_name.replace('_', ' ')}", fontsize=12, fontweight="bold", pad=12)
    bar_path = EXPLAIN_RESULTS_DIR / f"{dataset_prefix}_{model_name}_shap_bar.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "beeswarm_path": str(beeswarm_path),
        "bar_path": str(bar_path),
        "feature_names": feature_names
    }


def explain_single_prediction(
    student_dict: Dict[str, Any],
    model_name: str = "Random_Forest",
    dataset_prefix: str = "primary"
) -> Dict[str, Any]:
    """
    Generate local SHAP explanations and actionable plain-English insights for an individual student.
    """
    bundle = load_preprocessor_bundle(dataset_prefix)
    preprocessor = bundle["preprocessor"]
    feature_names = bundle["feature_names"]

    model_path = ML_MODELS_DIR / f"{dataset_prefix}_{model_name}.joblib"
    model = joblib.load(model_path)

    # Transform student vector
    df_student = pd.DataFrame([student_dict])
    X_single = preprocessor.transform(df_student)

    # Reference background sample
    from src.data.loader import load_dataset
    from src.data.preprocessing import get_feature_target_split
    csv_name = "student_data.csv" if dataset_prefix == "primary" else "university_student_retention_dataset_2134.csv"
    df_raw = load_dataset(csv_name)
    X_raw, _, _, _ = get_feature_target_split(df_raw, dataset_type=dataset_prefix)
    X_bg = preprocessor.transform(X_raw.head(50))

    explainer = get_model_explainer(model, X_bg)
    shap_explanation = explainer(X_single)

    if len(shap_explanation.shape) == 3 and shap_explanation.shape[2] == 2:
        student_shap = shap_explanation[0, :, 1]
    else:
        student_shap = shap_explanation[0]

    student_shap.feature_names = feature_names

    # Plot waterfall plot
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(student_shap, max_display=8, show=False)
    plt.title("Individual Prediction SHAP Waterfall Breakdown", fontsize=11, fontweight="bold", pad=10)
    waterfall_path = EXPLAIN_RESULTS_DIR / "single_waterfall.png"
    plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Extract top positive contributors (increasing risk) and negative contributors (reducing risk)
    contribs = []
    shap_values_arr = student_shap.values
    data_values_arr = student_shap.data

    for fname, s_val, d_val in zip(feature_names, shap_values_arr, data_values_arr):
        contribs.append({
            "feature": fname,
            "shap_value": round(float(s_val), 4),
            "feature_value": round(float(d_val), 2) if isinstance(d_val, (int, float, np.number)) else str(d_val),
            "direction": "Increases Dropout Risk" if s_val > 0 else "Protects Retention"
        })

    sorted_risk = sorted([c for c in contribs if c["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)
    sorted_protective = sorted([c for c in contribs if c["shap_value"] < 0], key=lambda x: x["shap_value"])

    # Generate plain-English explanations
    narrative_explanations = []
    if sorted_risk:
        top_r = sorted_risk[0]
        narrative_explanations.append(
            f"Primary Risk Driver: '{top_r['feature']}' heavily pushed the model toward predicting dropout (+{top_r['shap_value']:.2f} log-odds impact)."
        )
    if len(sorted_risk) > 1:
        sec_r = sorted_risk[1]
        narrative_explanations.append(
            f"Secondary Risk Factor: '{sec_r['feature']}' also contributed positively to the dropout risk assessment (+{sec_r['shap_value']:.2f})."
        )
    if sorted_protective:
        top_p = sorted_protective[0]
        narrative_explanations.append(
            f"Protective Buffer: '{top_p['feature']}' served as a strong retention shield (-{abs(top_p['shap_value']):.2f} reduction in risk)."
        )

    return {
        "waterfall_path": str(waterfall_path),
        "top_risk_factors": sorted_risk[:3],
        "top_protective_factors": sorted_protective[:3],
        "all_contributions": contribs,
        "narrative_explanations": narrative_explanations
    }
