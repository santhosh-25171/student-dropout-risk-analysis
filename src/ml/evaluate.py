"""
Machine Learning Model Evaluation and Comparison Engine.
Computes Accuracy, Precision, Recall, F1, ROC-AUC, CM, ROC/PR curves, and Feature Importance.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from src.config import ML_RESULTS_DIR, RANDOM_SEED


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """Calculate all standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: Path) -> None:
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Retained", "Dropout"],
        yticklabels=["Retained", "Dropout"]
    )
    ax.set_title(f"Confusion Matrix: {model_name.replace('_', ' ')}", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("True Class", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_roc_curves(model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]], y_true: np.ndarray, save_path: Path) -> None:
    """Plot combined ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.50)")
    
    colors = ["#2b5c8f", "#e67e22", "#27ae60", "#8e44ad", "#c0392b"]
    for idx, (name, (_, y_proba)) in enumerate(model_predictions.items()):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, label=f"{name.replace('_', ' ')} (AUC = {auc:.3f})", color=color, lw=2)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_pr_curves(model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]], y_true: np.ndarray, save_path: Path) -> None:
    """Plot combined Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 5))
    no_skill = (y_true == 1).mean()
    ax.plot([0, 1], [no_skill, no_skill], "k--", lw=1.2, label=f"Baseline Prevalance ({no_skill:.2f})")
    
    colors = ["#2b5c8f", "#e67e22", "#27ae60", "#8e44ad", "#c0392b"]
    for idx, (name, (_, y_proba)) in enumerate(model_predictions.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        color = colors[idx % len(colors)]
        ax.plot(recall, precision, label=name.replace('_', ' '), color=color, lw=2)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Dropout Detection Rate)", fontsize=11)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax.set_title("Precision-Recall Curve Comparison", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def extract_and_plot_feature_importance(
    model: Any, 
    model_name: str, 
    feature_names: List[str], 
    save_path: Path
) -> Optional[pd.DataFrame]:
    """Extract and plot feature importance / coefficients if supported."""
    importance_values = None
    metric_label = "Importance"

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
        metric_label = "Gini / Split Importance"
    elif hasattr(model, "coef_"):
        importance_values = model.coef_[0]
        metric_label = "Standardized Coefficient (Log-Odds Impact)"

    if importance_values is None or len(importance_values) != len(feature_names):
        return None

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": importance_values,
        "AbsValue": np.abs(importance_values)
    }).sort_values(by="AbsValue", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2b5c8f" if v >= 0 else "#d9534f" for v in fi_df["Value"]]
    ax.barh(fi_df["Feature"], fi_df["Value"], color=colors, height=0.6)
    ax.set_title(f"{model_name.replace('_', ' ')} — Feature Influences", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(metric_label, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    return fi_df.sort_values(by="AbsValue", ascending=False).drop(columns=["AbsValue"])


def evaluate_all_ml_models(
    trained_models: Dict[str, Any],
    data_bundle: Dict[str, Any],
    dataset_prefix: str = "primary"
) -> Dict[str, Any]:
    """
    Evaluates all 5 models on both validation and untouched test sets.
    Generates comparison tables, ROC curves, PR curves, CMs, and feature importance charts.
    Selects best model based on Recall and F1-score.
    """
    ML_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_val = data_bundle["X_val_proc"]
    y_val = data_bundle["y_val"]
    X_test = data_bundle["X_test_proc"]
    y_test = data_bundle["y_test"]
    feat_names = data_bundle["feature_names"]

    results = {
        "validation_metrics": {},
        "test_metrics": {},
        "best_model_name": None,
        "feature_importances": {}
    }

    test_predictions = {}
    comparison_rows = []

    for name, model in trained_models.items():
        # Validation Evaluation
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else y_val_pred
        val_m = compute_metrics(y_val, y_val_pred, y_val_proba)
        results["validation_metrics"][name] = val_m

        # Test Evaluation (Untouched partition)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_test_pred
        test_m = compute_metrics(y_test, y_test_pred, y_test_proba)
        results["test_metrics"][name] = test_m
        test_predictions[name] = (y_test_pred, y_test_proba)

        # Plot Confusion Matrix
        cm_path = ML_RESULTS_DIR / f"{dataset_prefix}_{name}_cm.png"
        plot_confusion_matrix(np.array(test_m["confusion_matrix"]), name, cm_path)

        # Feature Importance
        fi_path = ML_RESULTS_DIR / f"{dataset_prefix}_{name}_feat_imp.png"
        fi_df = extract_and_plot_feature_importance(model, name, feat_names, fi_path)
        if fi_df is not None:
            results["feature_importances"][name] = fi_df.to_dict(orient="records")

        # Collect comparison row
        comparison_rows.append({
            "Model": name.replace("_", " "),
            "Validation_Accuracy": val_m["accuracy"],
            "Validation_Recall": val_m["recall"],
            "Validation_F1": val_m["f1_score"],
            "Validation_ROC_AUC": val_m["roc_auc"],
            "Test_Accuracy": test_m["accuracy"],
            "Test_Precision": test_m["precision"],
            "Test_Recall": test_m["recall"],
            "Test_F1": test_m["f1_score"],
            "Test_ROC_AUC": test_m["roc_auc"]
        })

    # Combined ROC & PR Curves on Test Set
    roc_path = ML_RESULTS_DIR / f"{dataset_prefix}_roc_curves.png"
    plot_roc_curves(test_predictions, y_test, roc_path)

    pr_path = ML_RESULTS_DIR / f"{dataset_prefix}_pr_curves.png"
    plot_pr_curves(test_predictions, y_test, pr_path)

    # Save Comparison Data
    comp_df = pd.DataFrame(comparison_rows).sort_values(by="Test_F1", ascending=False)
    csv_path = ML_RESULTS_DIR / f"{dataset_prefix}_model_comparison.csv"
    comp_df.to_csv(csv_path, index=False)

    # Select best model: highest combined test recall and F1-score (critical for early warning)
    best_row = comp_df.iloc[0]
    best_name = best_row["Model"].replace(" ", "_")
    results["best_model_name"] = best_name
    results["best_model_summary"] = best_row.to_dict()

    json_path = ML_RESULTS_DIR / f"{dataset_prefix}_model_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
