"""
Deep Learning Training Engine for ANN and DNN.
Implements EarlyStopping, Learning Rate Scheduling, Class Weighting, and History Logging.
"""

from typing import Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import (
    RANDOM_SEED,
    DL_MODELS_DIR,
    DL_RESULTS_DIR
)
from src.data.loader import load_dataset
from src.data.preprocessing import (
    get_feature_target_split,
    create_train_val_test_splits,
    fit_and_transform_pipeline
)
from src.dl.ann import build_ann
from src.dl.dnn import build_dnn

# Set random seeds for reproducibility
tf.keras.utils.set_random_seed(RANDOM_SEED)


def train_deep_learning_models(
    dataset_name: str = "student_data.csv",
    epochs: int = 100,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Train both ANN and DNN models using identical leakage-safe splits.
    Saves trained .keras models and history JSON files.
    """
    DL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prefix = "primary" if "student_data" in dataset_name else "secondary"
    df = load_dataset(dataset_name)
    X, y, num_cols, cat_cols = get_feature_target_split(df, dataset_type=prefix)

    splits = create_train_val_test_splits(X, y, test_size=0.15, val_size=0.15, random_state=RANDOM_SEED)
    bundle = fit_and_transform_pipeline(splits, num_cols, cat_cols, dataset_name=prefix)

    X_train = bundle["X_train_proc"]
    y_train = bundle["y_train"].astype(np.float32)
    X_val = bundle["X_val_proc"]
    y_val = bundle["y_val"].astype(np.float32)
    class_weights = bundle["class_weights"]

    input_dim = X_train.shape[1]

    # Shared Callbacks
    def get_callbacks(name: str):
        return [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-5,
                verbose=0
            )
        ]

    # 1. Train ANN
    print(f"Training ANN on {dataset_name} (input_dim={input_dim})...")
    ann = build_ann(input_dim=input_dim)
    ann_hist = ann.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=get_callbacks("ANN"),
        verbose=0
    )
    ann_model_path = DL_MODELS_DIR / f"{prefix}_ann.keras"
    ann.save(ann_model_path)

    # 2. Train DNN
    print(f"Training DNN on {dataset_name} (input_dim={input_dim})...")
    dnn = build_dnn(input_dim=input_dim)
    dnn_hist = dnn.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=get_callbacks("DNN"),
        verbose=0
    )
    dnn_model_path = DL_MODELS_DIR / f"{prefix}_dnn.keras"
    dnn.save(dnn_model_path)

    # Convert histories to serializable dicts
    histories = {
        "ANN": {k: [float(v) for v in vals] for k, vals in ann_hist.history.items()},
        "DNN": {k: [float(v) for v in vals] for k, vals in dnn_hist.history.items()}
    }

    hist_path = DL_RESULTS_DIR / f"{prefix}_training_history.json"
    with open(hist_path, "w") as f:
        json.dump(histories, f, indent=2)

    return {
        "dataset_prefix": prefix,
        "ann_model": ann,
        "dnn_model": dnn,
        "histories": histories,
        "data_bundle": bundle
    }


def load_dl_model(model_type: str = "ann", dataset_prefix: str = "primary") -> keras.Model:
    """Load a trained Keras model."""
    path = DL_MODELS_DIR / f"{dataset_prefix}_{model_type.lower()}.keras"
    if not path.exists():
        raise FileNotFoundError(f"Deep learning model not found at {path}. Run DL training first.")
    return keras.models.load_model(path)
