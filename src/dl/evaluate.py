"""
Deep Learning Evaluation and ML vs DL Comparison Engine.
Generates learning curves, confusion matrices, and cross-paradigm benchmark charts.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
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
    classification_report
)

from src.config import DL_RESULTS_DIR, ML_RESULTS_DIR


def evaluate_keras_model(model: Any, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics for a Keras model."""
    probas = model.predict(X, verbose=0).flatten()
    preds = (probas >= threshold).astype(int)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    
    try:
        auc = roc_auc_score(y, probas)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y, preds)
    report = classification_report(y, preds, output_dict=True, zero_division=0)

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "probabilities": probas.tolist(),
        "predictions": preds.tolist()
    }


def plot_learning_curves(histories: Dict[str, Dict[str, List[float]]], save_dir: Path, prefix: str = "primary") -> Tuple[Path, Path]:
    """Plot training vs validation loss and accuracy graphs for both ANN and DNN."""
    # 1. Loss Graph
    fig_loss, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    for name, ax in zip(["ANN", "DNN"], [ax1, ax2]):
        hist = histories.get(name, {})
        if "loss" in hist and "val_loss" in hist:
            epochs = range(1, len(hist["loss"]) + 1)
            ax.plot(epochs, hist["loss"], "b-", lw=2, label="Train Loss")
            ax.plot(epochs, hist["val_loss"], "r--", lw=2, label="Validation Loss")
            ax.set_title(f"{name} — Loss Curve", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Binary Crossentropy Loss", fontsize=10)
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)

    loss_path = save_dir / f"{prefix}_loss_curves.png"
    fig_loss.tight_layout()
    fig_loss.savefig(loss_path, dpi=300)
    plt.close(fig_loss)

    # 2. Accuracy Graph
    fig_acc, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    for name, ax in zip(["ANN", "DNN"], [ax1, ax2]):
        hist = histories.get(name, {})
        if "accuracy" in hist and "val_accuracy" in hist:
            epochs = range(1, len(hist["accuracy"]) + 1)
            ax.plot(epochs, hist["accuracy"], "b-", lw=2, label="Train Accuracy")
            ax.plot(epochs, hist["val_accuracy"], "g--", lw=2, label="Validation Accuracy")
            ax.set_title(f"{name} — Accuracy Curve", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Accuracy", fontsize=10)
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)

    acc_path = save_dir / f"{prefix}_accuracy_curves.png"
    fig_acc.tight_layout()
    fig_acc.savefig(acc_path, dpi=300)
    plt.close(fig_acc)

    return loss_path, acc_path


def plot_dl_confusion_matrix(cm: np.ndarray, model_name: str, save_path: Path) -> None:
    """Plot and save confusion matrix for Deep Learning models."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax,
        xticklabels=["Retained", "Dropout"],
        yticklabels=["Retained", "Dropout"]
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("True Class", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def generate_ml_vs_dl_comparison(
    ml_results_path: Path, 
    dl_metrics: Dict[str, Dict[str, Any]], 
    save_dir: Path, 
    prefix: str = "primary"
) -> Tuple[pd.DataFrame, Path]:
    """Combine ML and DL test performance into a single benchmark table and comparison bar chart."""
    rows = []
    
    # Load ML results
    if ml_results_path.exists():
        with open(ml_results_path, "r") as f:
            ml_data = json.load(f)
            for m_name, metrics in ml_data.get("test_metrics", {}).items():
                rows.append({
                    "Paradigm": "Machine Learning",
                    "Model": m_name.replace("_", " "),
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1_Score": metrics["f1_score"],
                    "ROC_AUC": metrics["roc_auc"]
                })

    # Add DL results
    for m_name, metrics in dl_metrics.items():
        rows.append({
            "Paradigm": "Deep Learning",
            "Model": m_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1_Score": metrics["f1_score"],
            "ROC_AUC": metrics["roc_auc"]
        })

    df_comp = pd.DataFrame(rows).sort_values(by="F1_Score", ascending=False)
    csv_path = save_dir / f"{prefix}_ml_vs_dl_comparison.csv"
    df_comp.to_csv(csv_path, index=False)

    # Plot comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = df_comp.melt(id_vars=["Model", "Paradigm"], value_vars=["Recall", "F1_Score", "ROC_AUC"], 
                           var_name="Metric", value_name="Score")
    
    sns.barplot(data=df_plot, x="Model", y="Score", hue="Metric", palette="Set2", ax=ax)
    ax.set_title("Holistic Performance Comparison: Machine Learning vs Deep Learning", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_ylim([0.0, 1.05])
    plt.xticks(rotation=20, ha="right")
    ax.legend(loc="lower right")

    chart_path = save_dir / f"{prefix}_ml_vs_dl_comparison.png"
    fig.tight_layout()
    fig.savefig(chart_path, dpi=300)
    plt.close(fig)

    return df_comp, chart_path


def evaluate_deep_learning_models(
    dl_bundle: Dict[str, Any],
    dataset_prefix: str = "primary"
) -> Dict[str, Any]:
    """
    Evaluate ANN and DNN models, generate learning curves and ML vs DL comparison.
    """
    DL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ann = dl_bundle["ann_model"]
    dnn = dl_bundle["dnn_model"]
    histories = dl_bundle["histories"]
    data = dl_bundle["data_bundle"]

    X_val = data["X_val_proc"]
    y_val = data["y_val"]
    X_test = data["X_test_proc"]
    y_test = data["y_test"]

    # 1. Evaluate on Validation Set
    val_ann = evaluate_keras_model(ann, X_val, y_val)
    val_dnn = evaluate_keras_model(dnn, X_val, y_val)

    # 2. Evaluate on Test Set
    test_ann = evaluate_keras_model(ann, X_test, y_test)
    test_dnn = evaluate_keras_model(dnn, X_test, y_test)

    # 3. Plot Learning Curves
    plot_learning_curves(histories, DL_RESULTS_DIR, prefix=dataset_prefix)

    # 4. Plot DL Confusion Matrices
    plot_dl_confusion_matrix(np.array(test_ann["confusion_matrix"]), "ANN", DL_RESULTS_DIR / f"{dataset_prefix}_ANN_cm.png")
    plot_dl_confusion_matrix(np.array(test_dnn["confusion_matrix"]), "DNN", DL_RESULTS_DIR / f"{dataset_prefix}_DNN_cm.png")

    # 5. DL Comparison Table
    dl_table = [
        {
            "Model": "ANN (2 Dense Layers)",
            "Test_Accuracy": test_ann["accuracy"],
            "Test_Precision": test_ann["precision"],
            "Test_Recall": test_ann["recall"],
            "Test_F1": test_ann["f1_score"],
            "Test_ROC_AUC": test_ann["roc_auc"]
        },
        {
            "Model": "DNN (4 Layers + BatchNorm + Dropout)",
            "Test_Accuracy": test_dnn["accuracy"],
            "Test_Precision": test_dnn["precision"],
            "Test_Recall": test_dnn["recall"],
            "Test_F1": test_dnn["f1_score"],
            "Test_ROC_AUC": test_dnn["roc_auc"]
        }
    ]
    pd.DataFrame(dl_table).to_csv(DL_RESULTS_DIR / f"{dataset_prefix}_dl_comparison.csv", index=False)

    # 6. ML vs DL Cross-Comparison
    ml_comp_file = ML_RESULTS_DIR / f"{dataset_prefix}_model_comparison.json"
    dl_test_metrics = {"ANN": test_ann, "DNN": test_dnn}
    comp_df, _ = generate_ml_vs_dl_comparison(ml_comp_file, dl_test_metrics, DL_RESULTS_DIR, prefix=dataset_prefix)

    eval_summary = {
        "validation_metrics": {"ANN": val_ann, "DNN": val_dnn},
        "test_metrics": {"ANN": test_ann, "DNN": test_dnn},
        "ml_vs_dl_benchmark": comp_df.to_dict(orient="records")
    }

    with open(DL_RESULTS_DIR / f"{dataset_prefix}_dl_evaluation.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    return eval_summary
