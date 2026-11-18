"""
Leakage-safe data preprocessing pipelines, encoders, and train/val/test splitters.
"""

from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    TARGET_COLUMN,
    RANDOM_SEED,
    PRIMARY_EXCLUDE_COLS,
    SECONDARY_EXCLUDE_COLS,
    PREPROCESSING_DIR,
    PROCESSED_DATA_DIR
)


def get_feature_target_split(df: pd.DataFrame, dataset_type: str = "primary") -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Separate predictive features from target and leak/ID columns.
    Returns: (X, y, numerical_cols, categorical_cols)
    """
    exclude_cols = PRIMARY_EXCLUDE_COLS if dataset_type == "primary" else SECONDARY_EXCLUDE_COLS
    
    # Also drop any columns that contain string representations of lists or constant text
    drop_candidates = [col for col in exclude_cols + [TARGET_COLUMN] if col in df.columns]
    
    # In secondary dataset, drop serialised lists or unparseable columns if any
    if dataset_type != "primary":
        for col in ["gpa_by_semester", "attendance_by_semester"]:
            if col in df.columns and col not in drop_candidates:
                drop_candidates.append(col)

    X = df.drop(columns=drop_candidates).copy()
    y = df[TARGET_COLUMN].copy()

    # Drop any completely empty columns
    X = X.dropna(axis=1, how="all")

    # Determine numerical and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    return X, y, num_cols, cat_cols


def build_preprocessor(numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Build ColumnTransformer with median imputation + standard scaling and mode imputation + one-hot encoding."""
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformers = []
    if numerical_cols:
        transformers.append(("num", num_transformer, numerical_cols))
    if categorical_cols:
        transformers.append(("cat", cat_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def create_train_val_test_splits(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.15, 
    val_size: float = 0.15, 
    random_state: int = RANDOM_SEED
) -> Dict[str, Any]:
    """
    Create stratified 3-way split: Train (70%), Validation (15%), Test (15%).
    Strictly prevents data leakage by keeping validation and test isolated.
    """
    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Calculate relative validation size from train_val partition
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=random_state, stratify=y_train_val
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }


def fit_and_transform_pipeline(
    splits: Dict[str, Any], 
    num_cols: List[str], 
    cat_cols: List[str],
    dataset_name: str = "primary"
) -> Dict[str, Any]:
    """
    Fit preprocessor ONLY on training data, then transform train, val, and test.
    Saves preprocessor and transformed datasets to disk.
    """
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Fit exclusively on X_train
    X_train_proc = preprocessor.fit_transform(splits["X_train"])
    X_val_proc = preprocessor.transform(splits["X_val"])
    X_test_proc = preprocessor.transform(splits["X_test"])

    # Extract clean feature names
    feature_names = []
    try:
        raw_names = preprocessor.get_feature_names_out()
        feature_names = [n.replace("num__", "").replace("cat__", "") for n in raw_names]
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_train_proc.shape[1])]

    # Compute class weights for training
    classes = np.unique(splits["y_train"])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=splits["y_train"])
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}

    # Package processed data
    processed_bundle = {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "raw_numerical_cols": num_cols,
        "raw_categorical_cols": cat_cols,
        "class_weights": class_weight_dict,
        "X_train_raw": splits["X_train"],
        "y_train": splits["y_train"].values,
        "X_val_raw": splits["X_val"],
        "y_val": splits["y_val"].values,
        "X_test_raw": splits["X_test"],
        "y_test": splits["y_test"].values,
        "X_train_proc": X_train_proc,
        "X_val_proc": X_val_proc,
        "X_test_proc": X_test_proc
    }

    # Save artifacts
    save_path = PREPROCESSING_DIR / f"{dataset_name}_preprocessor.joblib"
    joblib.dump(preprocessor, save_path)

    meta_path = PREPROCESSING_DIR / f"{dataset_name}_meta.joblib"
    joblib.dump({
        "feature_names": feature_names,
        "numerical_cols": num_cols,
        "categorical_cols": cat_cols,
        "class_weights": class_weight_dict
    }, meta_path)

    return processed_bundle


def load_preprocessor_bundle(dataset_name: str = "primary") -> Dict[str, Any]:
    """Load fitted preprocessor and metadata."""
    prep_path = PREPROCESSING_DIR / f"{dataset_name}_preprocessor.joblib"
    meta_path = PREPROCESSING_DIR / f"{dataset_name}_meta.joblib"
    
    if not prep_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Preprocessor bundle for '{dataset_name}' not found. Run preprocessing first.")
        
    preprocessor = joblib.load(prep_path)
    meta = joblib.load(meta_path)
    return {
        "preprocessor": preprocessor,
        **meta
    }
