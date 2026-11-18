"""
EDUVISION AI — Master Pipeline Runner CLI.
Allows running individual modules or end-to-end training and evaluation independently.

Usage:
  python run_pipeline.py --inspect
  python run_pipeline.py --eda
  python run_pipeline.py --train-ml
  python run_pipeline.py --train-dl
  python run_pipeline.py --xai
  python run_pipeline.py --reports
  python run_pipeline.py --all
"""

import argparse
import sys
import json
from pathlib import Path

from src.config import PRIMARY_DATASET_NAME, SECONDARY_DATASET_NAME
from src.data.loader import load_dataset, get_dataset_summary
from src.data.validation import compare_dataset_schemas
from src.eda.visualizations import generate_all_eda_visualizations
from src.ml.train import train_all_ml_models
from src.ml.evaluate import evaluate_all_ml_models
from src.dl.train import train_deep_learning_models
from src.dl.evaluate import evaluate_deep_learning_models
from src.explainability.shap_analysis import compute_global_shap
from src.reports.export import generate_executive_markdown_report, save_project_report


def run_inspect():
    print("=" * 60)
    print("🔍 STEP 1: DATASET INSPECTION & COMPATIBILITY AUDIT")
    print("=" * 60)
    
    df1 = load_dataset(PRIMARY_DATASET_NAME)
    s1 = get_dataset_summary(df1)
    print(f"\n[Primary Dataset: {PRIMARY_DATASET_NAME}]")
    print(f" - Records: {s1['total_records']}, Features: {s1['total_columns']}")
    print(f" - Missing cells: {s1['total_missing_cells']}, Duplicates: {s1['duplicate_rows']}")
    print(f" - Target Distribution: {s1['target_distribution']}")

    df2 = load_dataset(SECONDARY_DATASET_NAME)
    s2 = get_dataset_summary(df2)
    print(f"\n[Secondary Dataset: {SECONDARY_DATASET_NAME}]")
    print(f" - Records: {s2['total_records']}, Features: {s2['total_columns']}")
    print(f" - Missing cells: {s2['total_missing_cells']}, Duplicates: {s2['duplicate_rows']}")

    comp = compare_dataset_schemas(df1, df2)
    print(f"\n[Schema Compatibility]: {comp['overlap_percentage']}% overlap.")
    print(f"Recommendation: {comp['recommendation']}")


def run_eda():
    print("=" * 60)
    print("📊 STEP 2: GENERATING PUBLICATION-QUALITY EDA CHARTS")
    print("=" * 60)
    df = load_dataset(PRIMARY_DATASET_NAME)
    paths = generate_all_eda_visualizations(df)
    for name, p in paths.items():
        print(f" - Saved {name}: {p}")
    print("EDA generation complete.")


def run_ml():
    print("=" * 60)
    print("🤖 STEP 3: TRAINING & EVALUATING 5 ML ALGORITHMS")
    print("=" * 60)
    print("Training on primary dataset...")
    train_out = train_all_ml_models(PRIMARY_DATASET_NAME)
    eval_out = evaluate_all_ml_models(train_out["trained_models"], train_out["data_bundle"], train_out["dataset_prefix"])
    print(f"Best ML Model: {eval_out['best_model_name']}")
    print(f"Best Model Test Metrics: {eval_out['best_model_summary']}")

    print("\nTraining on secondary dataset (institutional)...")
    train_out2 = train_all_ml_models(SECONDARY_DATASET_NAME)
    evaluate_all_ml_models(train_out2["trained_models"], train_out2["data_bundle"], train_out2["dataset_prefix"])
    print("ML pipeline complete.")


def run_dl(epochs=60):
    print("=" * 60)
    print("🧠 STEP 4: TRAINING & EVALUATING DEEP LEARNING (ANN & DNN)")
    print("=" * 60)
    dl_bundle = train_deep_learning_models(PRIMARY_DATASET_NAME, epochs=epochs, batch_size=32)
    dl_eval = evaluate_deep_learning_models(dl_bundle, "primary")
    print(f"ANN Test F1: {dl_eval['test_metrics']['ANN']['f1_score']:.4f}")
    print(f"DNN Test Recall: {dl_eval['test_metrics']['DNN']['recall']:.4f}")
    print("DL pipeline complete.")


def run_xai():
    print("=" * 60)
    print("💡 STEP 5: EXPLAINABLE AI (SHAP GLOBAL ATTRIBUTION)")
    print("=" * 60)
    res = compute_global_shap("Random_Forest", "primary", max_samples=75)
    print(f"Saved Beeswarm: {res['beeswarm_path']}")
    print(f"Saved Bar Importance: {res['bar_path']}")
    print("XAI generation complete.")


def run_reports():
    print("=" * 60)
    print("📑 STEP 6: GENERATING EXECUTIVE ANALYSIS REPORT")
    print("=" * 60)
    from src.config import ML_RESULTS_DIR, DL_RESULTS_DIR
    
    df1 = load_dataset(PRIMARY_DATASET_NAME)
    s1 = get_dataset_summary(df1)

    ml_file = ML_RESULTS_DIR / "primary_model_comparison.json"
    ml_data = json.load(open(ml_file)) if ml_file.exists() else {}

    dl_file = DL_RESULTS_DIR / "primary_dl_evaluation.json"
    dl_data = json.load(open(dl_file)) if dl_file.exists() else {}

    report_md = generate_executive_markdown_report(ml_data, dl_data, s1)
    saved_path = save_project_report(report_md)
    print(f"Executive Report successfully written to: {saved_path}")


def main():
    parser = argparse.ArgumentParser(description="EDUVISION AI Master Runner")
    parser.add_argument("--inspect", action="store_true", help="Inspect datasets and schema compatibility")
    parser.add_argument("--eda", action="store_true", help="Generate all EDA figures")
    parser.add_argument("--train-ml", action="store_true", help="Train and tune 5 ML algorithms")
    parser.add_argument("--train-dl", action="store_true", help="Train ANN and DNN")
    parser.add_argument("--xai", action="store_true", help="Generate SHAP global plots")
    parser.add_argument("--reports", action="store_true", help="Generate executive report")
    parser.add_argument("--all", action="store_true", help="Run full pipeline end-to-end")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.all:
        run_inspect()
        run_eda()
        run_ml()
        run_dl()
        run_xai()
        run_reports()
    else:
        if args.inspect: run_inspect()
        if args.eda: run_eda()
        if args.train_ml: run_ml()
        if args.train_dl: run_dl()
        if args.xai: run_xai()
        if args.reports: run_reports()


if __name__ == "__main__":
    main()
