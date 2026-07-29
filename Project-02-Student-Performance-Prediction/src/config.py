"""
Configuration Module
Centralizes all configuration settings for the Student Performance Prediction System.
"""

import os
from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET_PATH = PROCESSED_DATA_DIR / "student_performance.csv"

# Model Paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "student_performance_model.pkl"

# Output Paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Logging Paths
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "application.log"

# Source and Tests
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"

# Application Constants
APP_NAME = "Student Performance Prediction System"
APP_VERSION = "1.0.0"
AUTHOR = "CodeVedX Intern"

# Model Configuration
MODEL_FEATURES = [
    'Hours_Studied',
    'Attendance',
    'Parental_Involvement',
    'Access_to_Resources',
    'Extracurricular_Activities',
    'Sleep_Hours',
    'Previous_Scores',
    'Motivation_Level',
    'Internet_Access',
    'Tutoring_Sessions',
    'Family_Income',
    'Teacher_Quality',
    'School_Type',
    'Peer_Influence',
    'Physical_Activity',
    'Learning_Disabilities',
    'Parental_Education_Level',
    'Distance_from_Home',
    'Gender'
]

# Validation Ranges
VALIDATION_RANGES = {
    'Hours_Studied': {'min': 0, 'max': 50},
    'Attendance': {'min': 0, 'max': 100},
    'Sleep_Hours': {'min': 0, 'max': 24},
    'Previous_Scores': {'min': 0, 'max': 100},
    'Tutoring_Sessions': {'min': 0, 'max': 10},
    'Physical_Activity': {'min': 0, 'max': 10}
}

# Categorical Values
CATEGORICAL_VALUES = {
    'Gender': ['Male', 'Female'],
    'School_Type': ['Public', 'Private'],
    'Teacher_Quality': ['Low', 'Medium', 'High'],
    'Parental_Involvement': ['Low', 'Medium', 'High'],
    'Access_to_Resources': ['Low', 'Medium', 'High'],
    'Extracurricular_Activities': ['Yes', 'No'],
    'Motivation_Level': ['Low', 'Medium', 'High'],
    'Internet_Access': ['Yes', 'No'],
    'Family_Income': ['Low', 'Medium', 'High'],
    'Peer_Influence': ['Negative', 'Neutral', 'Positive'],
    'Learning_Disabilities': ['Yes', 'No'],
    'Parental_Education_Level': ['High School', 'College', 'Postgraduate'],
    'Distance_from_Home': ['Near', 'Moderate', 'Far']
}

# Menu Options
MENU_OPTIONS = {
    1: "View Dataset Information",
    2: "Search Student",
    3: "Add Student Record",
    4: "Update Student Record",
    5: "Delete Student Record",
    6: "Predict Student Performance",
    7: "Batch Prediction",
    8: "Export Predictions",
    9: "View Model Information",
    10: "Exit"
}


def create_directories():
    """
    Create all necessary directories if they don't exist.
    """
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        PREDICTIONS_DIR,
        CHARTS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
        SRC_DIR,
        TESTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Ensure directories exist on import
create_directories()