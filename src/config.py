"""
Configuration constants and global paths for EDUVISION AI.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
ML_MODELS_DIR = MODELS_DIR / "ml"
DL_MODELS_DIR = MODELS_DIR / "dl"
PREPROCESSING_DIR = MODELS_DIR / "preprocessing"

RESULTS_DIR = BASE_DIR / "results"
EDA_RESULTS_DIR = RESULTS_DIR / "eda"
ML_RESULTS_DIR = RESULTS_DIR / "ml"
DL_RESULTS_DIR = RESULTS_DIR / "dl"
EXPLAIN_RESULTS_DIR = RESULTS_DIR / "explainability"

REPORTS_DIR = RESULTS_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
PRIMARY_DATASET_NAME = "student_data.csv"
SECONDARY_DATASET_NAME = "university_student_retention_dataset_2134.csv"

# Global Seed
RANDOM_SEED = 42

# Target Variable
TARGET_COLUMN = "next_semester_dropout"

# Positive Class Definition: 1 indicates Student Dropped Out (At Risk), 0 indicates Retained
POSITIVE_CLASS = 1

# Leaky/Post-hoc and ID columns to exclude from training
PRIMARY_EXCLUDE_COLS = ["student_id"]
SECONDARY_EXCLUDE_COLS = [
    "student_id",
    "dropout_risk",       # Post-hoc outcome proxy
    "dropout_type",       # Recorded after dropout
    "dropout_duration",   # Recorded after dropout
    "dropout_cause",      # Recorded after dropout
    "advising_notes",     # Single constant string across rows
    "counseling_notes",   # Constant or null string
]

# Risk Thresholds
DEFAULT_RISK_THRESHOLDS = {
    "low_max": 0.30,
    "medium_max": 0.65
}

# UI Theme Config
APP_TITLE = "EDUVISION AI — Intelligent Student Dropout Prediction"
APP_SUBTITLE = "Early Intervention & Academic Decision-Support System"
