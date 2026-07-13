"""
Configuration Module for Utility Usage Prediction Tool

This module contains all configuration settings including:
- Dataset paths
- Model paths
- Output paths
- Logging paths
- Application constants

Author: CodeVedX AI/ML Internship
"""

from pathlib import Path
from typing import Dict, Any

# ========================================
# PROJECT PATHS
# ========================================

# Project root directory (parent of src/)
PROJECT_ROOT: Path = Path(__file__).parent.parent

# ========================================
# DATA PATHS
# ========================================

# Raw data directory
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"

# Processed data directory
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"

# Main dataset file
DATASET_PATH: Path = PROCESSED_DATA_DIR / "utility_usage.csv"

# ========================================
# MODEL PATHS
# ========================================

# Models directory
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Model file path
MODEL_PATH: Path = MODELS_DIR / "utility_usage_model.pkl"

# ========================================
# OUTPUT PATHS
# ========================================

# Main outputs directory
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

# Charts directory
CHARTS_DIR: Path = OUTPUTS_DIR / "charts"

# Predictions directory
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"

# Reports directory
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"

# ========================================
# LOGGING PATHS
# ========================================

# Logs directory
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Application log file
LOG_FILE: Path = LOGS_DIR / "application.log"

# ========================================
# APPLICATION CONSTANTS
# ========================================

# Application Information
APP_NAME: str = "Utility Usage Prediction Tool"
APP_VERSION: str = "1.0.0"
APP_AUTHOR: str = "CodeVedX AI/ML Internship"

# Model Configuration
MODEL_TYPE: str = "Linear Regression"
TARGET_COLUMN: str = "Global_active_power"
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# Feature Columns for Prediction
FEATURE_COLUMNS: list[str] = [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "Year",
    "Month",
    "Day",
    "Hour"
]

# CSV Configuration
CSV_DELIMITER: str = ","
CSV_ENCODING: str = "utf-8"

# Validation Ranges
MIN_VOLTAGE: float = 200.0
MAX_VOLTAGE: float = 250.0
MIN_INTENSITY: float = 0.0
MAX_INTENSITY: float = 50.0
MIN_ACTIVE_POWER: float = 0.0
MAX_ACTIVE_POWER: float = 10.0

# Menu Configuration
MENU_OPTIONS: Dict[int, str] = {
    1: "Add Record",
    2: "View Records",
    3: "Search Record",
    4: "Update Record",
    5: "Delete Record",
    6: "Train ML Model",
    7: "Predict Usage",
    8: "View Model Metrics",
    9: "Export Prediction Report",
    10: "Exit"
}

# Display Configuration
MAX_RECORDS_DISPLAY: int = 100
TABLE_WIDTH: int = 120

# ========================================
# HELPER FUNCTIONS
# ========================================


def ensure_directories() -> None:
    """
    Create all required directories if they don't exist.
    
    This function ensures that all project directories are created
    before the application runs.
    """
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        CHARTS_DIR,
        PREDICTIONS_DIR,
        REPORTS_DIR,
        LOGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_config_dict() -> Dict[str, Any]:
    """
    Get all configuration as a dictionary.
    
    Returns:
        Dictionary containing all configuration values
    """
    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "dataset_path": str(DATASET_PATH),
        "model_path": str(MODEL_PATH),
        "logs_path": str(LOG_FILE),
        "model_type": MODEL_TYPE,
        "target_column": TARGET_COLUMN,
        "features": FEATURE_COLUMNS
    }


# ========================================
# INITIALIZATION
# ========================================

# Create directories on import
ensure_directories()