"""
Configuration settings for the Fake News Detection Tool.
Centralizes all paths, settings, and application constants.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "fake_news_dataset.csv"

# Model paths
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts"
REPORTS_DIR = OUTPUT_DIR / "reports"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

# Log paths
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "application.log"

# Application settings
APP_NAME = "AI Based Fake News Detection Tool"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "A machine learning-based tool for detecting fake news articles"

# Model information
MODEL_INFO = {
    "name": "Random Forest",
    "algorithm": "Random Forest Classifier",
    "training_dataset": "Fake News Dataset (Kaggle)",
    "training_samples": 35176,
    "test_samples": 8795,
    "accuracy": 0.9968,
    "precision": 0.9948,
    "recall": 0.9986,
    "f1_score": 0.9967,
    "roc_auc": 0.9999,
    "vectorizer": "TF-IDF",
    "vocabulary_size": 20000,
    "ngram_range": "(1, 2)",
    "training_date": "2024",
}

# Prediction settings
PREDICTION_HISTORY_FILE = PREDICTIONS_DIR / "prediction_history.csv"
MAX_HISTORY_DISPLAY = 50

# Validation settings
MIN_TEXT_LENGTH = 20
MAX_TEXT_LENGTH = 50000

# Ensure directories exist
for directory in [LOG_DIR, PREDICTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)