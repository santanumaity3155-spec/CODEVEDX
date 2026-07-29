"""
Tests for Configuration Module
"""

import pytest
from pathlib import Path
from src.config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR,
    DATASET_PATH, MODEL_PATH, PREDICTIONS_DIR, LOG_FILE,
    APP_NAME, APP_VERSION, MODEL_FEATURES, VALIDATION_RANGES,
    CATEGORICAL_VALUES, MENU_OPTIONS, create_directories
)


class TestConfigPaths:
    """Test configuration paths."""
    
    def test_project_root_exists(self):
        """Test that project root is defined correctly."""
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()
    
    def test_data_directories(self):
        """Test that data directories are defined."""
        assert DATA_DIR.name == "data"
        assert MODELS_DIR.name == "models"
    
    def test_dataset_path(self):
        """Test that dataset path is correctly formed."""
        assert DATASET_PATH.name == "student_performance.csv"
        assert DATASET_PATH.parent.name == "processed"
    
    def test_model_path(self):
        """Test that model path is correctly formed."""
        assert MODEL_PATH.name == "student_performance_model.pkl"
        assert MODEL_PATH.parent.name == "models"
    
    def test_output_directories(self):
        """Test that output directories are defined."""
        assert OUTPUTS_DIR.name == "outputs"
        assert PREDICTIONS_DIR.name == "predictions"
        assert PREDICTIONS_DIR.parent == OUTPUTS_DIR
    
    def test_log_path(self):
        """Test that log path is correctly formed."""
        assert LOG_FILE.name == "application.log"
        assert LOG_FILE.parent.name == "logs"


class TestConfigConstants:
    """Test configuration constants."""
    
    def test_app_name(self):
        """Test that app name is defined."""
        assert APP_NAME == "Student Performance Prediction System"
        assert isinstance(APP_NAME, str)
    
    def test_app_version(self):
        """Test that app version is defined."""
        assert APP_VERSION == "1.0.0"
        assert isinstance(APP_VERSION, str)
    
    def test_model_features(self):
        """Test that model features list is defined."""
        assert isinstance(MODEL_FEATURES, list)
        assert len(MODEL_FEATURES) > 0
        assert 'Hours_Studied' in MODEL_FEATURES
        assert 'Attendance' in MODEL_FEATURES
    
    def test_validation_ranges(self):
        """Test that validation ranges are defined."""
        assert isinstance(VALIDATION_RANGES, dict)
        assert 'Hours_Studied' in VALIDATION_RANGES
        assert 'Attendance' in VALIDATION_RANGES
        
        # Check structure
        for field, range_dict in VALIDATION_RANGES.items():
            assert 'min' in range_dict
            assert 'max' in range_dict
            assert range_dict['min'] < range_dict['max']
    
    def test_categorical_values(self):
        """Test that categorical values are defined."""
        assert isinstance(CATEGORICAL_VALUES, dict)
        assert 'Gender' in CATEGORICAL_VALUES
        assert 'School_Type' in CATEGORICAL_VALUES
        assert 'Teacher_Quality' in CATEGORICAL_VALUES
        
        # Check that values are lists
        for field, values in CATEGORICAL_VALUES.items():
            assert isinstance(values, list)
            assert len(values) > 0
    
    def test_menu_options(self):
        """Test that menu options are defined."""
        assert isinstance(MENU_OPTIONS, dict)
        assert len(MENU_OPTIONS) == 10
        
        # Check that all options are present
        for i in range(1, 11):
            assert i in MENU_OPTIONS
            assert isinstance(MENU_OPTIONS[i], str)
            assert len(MENU_OPTIONS[i]) > 0


class TestCreateDirectories:
    """Test directory creation function."""
    
    def test_create_directories_runs(self):
        """Test that create_directories function runs without errors."""
        # This should not raise an exception
        create_directories()
        
        # Verify directories exist
        assert DATA_DIR.exists()
        assert MODELS_DIR.exists()
        assert OUTPUTS_DIR.exists()
        assert LOGS_DIR.exists()
        assert PREDICTIONS_DIR.exists()
    
    def test_create_directories_idempotent(self):
        """Test that create_directories can be called multiple times."""
        # Should not raise an exception on second call
        create_directories()
        create_directories()
        
        # Directories should still exist
        assert DATA_DIR.exists()
        assert MODELS_DIR.exists()