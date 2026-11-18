"""
Machine Learning Training Engine.
Implements 5 tuned algorithms with 5-fold Stratified Cross-Validation on training data.
"""

from typing import Dict, Any, Tuple
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from src.config import (
    RANDOM_SEED,
    ML_MODELS_DIR,
    ML_RESULTS_DIR
)
from src.data.loader import load_dataset
from src.data.preprocessing import (
    get_feature_target_split,
    create_train_val_test_splits,
    fit_and_transform_pipeline
)


def get_model_definitions() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Returns the 5 required algorithms and hyperparameter search grids.
    All models with class_weight support are set to 'balanced' to handle dropout class imbalance.
    """
    models = {
        "Logistic_Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["liblinear", "lbfgs"]
            }
        ),
        "Decision_Tree": (
            DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
            {
                "max_depth": [3, 5, 8, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        ),
        "Random_Forest": (
            RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 8, 12, None],
                "min_samples_split": [2, 5]
            }
        ),
        "Support_Vector_Machine": (
            SVC(probability=True, random_state=RANDOM_SEED, class_weight="balanced"),
            {
                "C": [0.1, 1.0, 5.0],
                "kernel": ["linear", "rbf"]
            }
        ),
        "Gradient_Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_SEED),
            {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 4, 5]
            }
        )
    }
    return models


def train_and_tune_model(
    model_name: str,
    base_model: Any,
    param_grid: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Tuple[Any, Dict[str, Any], float]:
    """Train and tune a model using 5-fold Stratified K-Fold CV scoring for ROC-AUC."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_all_ml_models(dataset_name: str = "student_data.csv") -> Dict[str, Any]:
    """
    Executes end-to-end ML pipeline:
    1. Loads dataset.
    2. Splits into Train (70%), Validation (15%), Test (15%).
    3. Fits preprocessor strictly on training data.
    4. Tunes and fits all 5 algorithms using 5-fold Stratified CV.
    5. Saves fitted models and metadata.
    """
    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ML_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prefix = "primary" if "student_data" in dataset_name else "secondary"
    df = load_dataset(dataset_name)
    X, y, num_cols, cat_cols = get_feature_target_split(df, dataset_type=prefix)

    # 3-way split
    splits = create_train_val_test_splits(X, y, test_size=0.15, val_size=0.15, random_state=RANDOM_SEED)

    # Fit preprocessor strictly on training data
    data_bundle = fit_and_transform_pipeline(splits, num_cols, cat_cols, dataset_name=prefix)
    X_tr = data_bundle["X_train_proc"]
    y_tr = data_bundle["y_train"]

    model_defs = get_model_definitions()
    trained_models = {}
    cv_summary = {}

    for name, (base_clf, grid) in model_defs.items():
        best_clf, best_params, best_cv_auc = train_and_tune_model(name, base_clf, grid, X_tr, y_tr)
        
        # Save model
        save_file = ML_MODELS_DIR / f"{prefix}_{name}.joblib"
        joblib.dump(best_clf, save_file)

        trained_models[name] = best_clf
        cv_summary[name] = {
            "best_params": best_params,
            "cv_roc_auc_score": round(float(best_cv_auc), 4),
            "model_path": str(save_file)
        }

    # Save CV summary
    summary_path = ML_RESULTS_DIR / f"{prefix}_cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(cv_summary, f, indent=2)

    return {
        "dataset_prefix": prefix,
        "trained_models": trained_models,
        "cv_summary": cv_summary,
        "data_bundle": data_bundle
    }


def load_ml_model(model_name: str, dataset_prefix: str = "primary") -> Any:
    """Load a trained ML model by algorithm name."""
    model_path = ML_MODELS_DIR / f"{dataset_prefix}_{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file '{model_path}' not found. Run ML training first.")
    return joblib.load(model_path)
