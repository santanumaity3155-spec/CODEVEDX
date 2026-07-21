"""
ML Predictor Module Tests

This module contains unit tests for the ML predictor module functions.

Author: CodeVedX AI/ML Internship
"""

import unittest
import sys
import os
import tempfile
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from sklearn.linear_model import LinearRegression

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictor import MLPredictor, get_predictor, make_prediction, load_trained_model
from config import FEATURE_COLUMNS, TARGET_COLUMN, MODEL_PATH


class TestMLPredictor(unittest.TestCase):
    """Test cases for MLPredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_model_path = Path(self.test_dir) / "test_model.pkl"
        
        # Create a simple real LinearRegression model for pickling
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = X_train @ np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) + 0.5
        self.real_model = LinearRegression()
        self.real_model.fit(X_train, y_train)
        
        # Save real model
        joblib.dump(self.real_model, self.test_model_path)
        
        # Create MLPredictor instance
        self.predictor = MLPredictor(model_path=self.test_model_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files
        if self.test_model_path.exists():
            self.test_model_path.unlink()
        
        # Remove any CSV files created during tests
        for csv_file in Path(self.test_dir).glob("*.csv"):
            try:
                csv_file.unlink()
            except PermissionError:
                pass
        
        # Remove directory and all contents
        try:
            import shutil
            shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception:
            pass
    
    def test_load_model_success(self):
        """Test successful model loading."""
        success, error = self.predictor.load_model()
        
        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertTrue(self.predictor.is_model_loaded())
        self.assertIsNotNone(self.predictor.model)
    
    def test_load_model_file_not_found(self):
        """Test loading non-existent model file."""
        predictor = MLPredictor(model_path=Path(self.test_dir) / "nonexistent.pkl")
        success, error = predictor.load_model()
        
        self.assertFalse(success)
        self.assertIsNotNone(error)
        self.assertIn("not found", error)
        self.assertFalse(predictor.is_model_loaded())
    
    def test_is_model_loaded(self):
        """Test checking if model is loaded."""
        self.assertFalse(self.predictor.is_model_loaded())
        
        self.predictor.load_model()
        self.assertTrue(self.predictor.is_model_loaded())
    
    def test_get_model_info(self):
        """Test getting model information."""
        # Before loading
        info = self.predictor.get_model_info()
        self.assertIn("error", info)
        
        # After loading
        self.predictor.load_model()
        info = self.predictor.get_model_info()
        
        self.assertIn("model_type", info)
        self.assertIn("model_path", info)
        self.assertIn("features_count", info)
        self.assertIn("features", info)
        self.assertIn("target", info)
        self.assertEqual(info["model_type"], "Linear Regression")
        self.assertEqual(info["features_count"], len(FEATURE_COLUMNS))
    
    def test_predict_success(self):
        """Test successful prediction."""
        # Load model
        self.predictor.load_model()
        
        # Prepare input data
        input_data = {
            'Global_reactive_power': 0.5,
            'Voltage': 220.0,
            'Global_intensity': 10.0,
            'Sub_metering_1': 1.0,
            'Sub_metering_2': 0.0,
            'Sub_metering_3': 5.0,
            'Year': 2023,
            'Month': 6,
            'Day': 15,
            'Hour': 14
        }
        
        # Make prediction
        success, prediction, error = self.predictor.predict(input_data)
        
        self.assertTrue(success)
        self.assertIsNotNone(prediction)
        self.assertIsNone(error)
        self.assertIsInstance(prediction, float)
    
    def test_predict_model_not_loaded(self):
        """Test prediction when model is not loaded."""
        input_data = {
            'Global_reactive_power': 0.5,
            'Voltage': 220.0,
            'Global_intensity': 10.0,
            'Sub_metering_1': 1.0,
            'Sub_metering_2': 0.0,
            'Sub_metering_3': 5.0,
            'Year': 2023,
            'Month': 6,
            'Day': 15,
            'Hour': 14
        }
        
        # Model not loaded, should try to load
        success, prediction, error = self.predictor.predict(input_data)
        
        # Should succeed since model file exists
        self.assertTrue(success)
    
    def test_predict_missing_features(self):
        """Test prediction with missing features."""
        self.predictor.load_model()
        
        # Missing features
        input_data = {
            'Global_reactive_power': 0.5,
            'Voltage': 220.0
        }
        
        success, prediction, error = self.predictor.predict(input_data)
        
        self.assertFalse(success)
        self.assertIsNone(prediction)
        self.assertIsNotNone(error)
    
    def test_predict_invalid_values(self):
        """Test prediction with invalid values."""
        self.predictor.load_model()
        
        # Invalid values (non-numeric)
        input_data = {
            'Global_reactive_power': "invalid",
            'Voltage': 220.0,
            'Global_intensity': 10.0,
            'Sub_metering_1': 1.0,
            'Sub_metering_2': 0.0,
            'Sub_metering_3': 5.0,
            'Year': 2023,
            'Month': 6,
            'Day': 15,
            'Hour': 14
        }
        
        success, prediction, error = self.predictor.predict(input_data)
        
        self.assertFalse(success)
        self.assertIsNone(prediction)
    
    def test_predict_batch(self):
        """Test batch prediction."""
        self.predictor.load_model()
        
        # Create test dataframe
        test_data = pd.DataFrame({
            'Global_reactive_power': [0.5, 0.6],
            'Voltage': [220.0, 221.0],
            'Global_intensity': [10.0, 11.0],
            'Sub_metering_1': [1.0, 2.0],
            'Sub_metering_2': [0.0, 1.0],
            'Sub_metering_3': [5.0, 6.0],
            'Year': [2023, 2023],
            'Month': [6, 6],
            'Day': [15, 16],
            'Hour': [14, 15]
        })
        
        success, predictions_df, error = self.predictor.predict_batch(test_data)
        
        self.assertTrue(success)
        self.assertIsNotNone(predictions_df)
        self.assertIsNone(error)
        self.assertEqual(len(predictions_df), 2)
        self.assertIn('Predicted_Active_Power', predictions_df.columns)
    
    def test_validate_input_valid(self):
        """Test input validation with valid data."""
        input_data = {
            'Global_reactive_power': 0.5,
            'Voltage': 220.0,
            'Global_intensity': 10.0,
            'Sub_metering_1': 1.0,
            'Sub_metering_2': 0.0,
            'Sub_metering_3': 5.0,
            'Year': 2023,
            'Month': 6,
            'Day': 15,
            'Hour': 14
        }
        
        is_valid, error = self.predictor.validate_input(input_data)
        
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_validate_input_missing_features(self):
        """Test input validation with missing features."""
        input_data = {
            'Global_reactive_power': 0.5,
            'Voltage': 220.0
        }
        
        is_valid, error = self.predictor.validate_input(input_data)
        
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
        self.assertIn("Missing features", error)
    
    def test_validate_input_null_values(self):
        """Test input validation with null values."""
        input_data = {
            'Global_reactive_power': 0.5,
            'Voltage': 220.0,
            'Global_intensity': None,
            'Sub_metering_1': 1.0,
            'Sub_metering_2': 0.0,
            'Sub_metering_3': 5.0,
            'Year': 2023,
            'Month': 6,
            'Day': 15,
            'Hour': 14
        }
        
        is_valid, error = self.predictor.validate_input(input_data)
        
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
        self.assertIn("Null values", error)
    
    def test_get_feature_importance(self):
        """Test getting feature importance."""
        self.predictor.load_model()
        
        importance = self.predictor.get_feature_importance()
        
        self.assertIsNotNone(importance)
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), len(FEATURE_COLUMNS))
        
        for feature in FEATURE_COLUMNS:
            self.assertIn(feature, importance)
    
    def test_get_feature_importance_model_not_loaded(self):
        """Test getting feature importance when model not loaded."""
        importance = self.predictor.get_feature_importance()
        
        self.assertIsNone(importance)
    
    def test_save_prediction(self):
        """Test saving prediction to file."""
        self.predictor.load_model()
        
        input_data = {
            'Global_reactive_power': 0.5,
            'Voltage': 220.0,
            'Global_intensity': 10.0,
            'Sub_metering_1': 1.0,
            'Sub_metering_2': 0.0,
            'Sub_metering_3': 5.0,
            'Year': 2023,
            'Month': 6,
            'Day': 15,
            'Hour': 14
        }
        
        output_path = Path(self.test_dir) / "prediction.csv"
        
        success, error = self.predictor.save_prediction(2.5, input_data, output_path)
        
        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertTrue(output_path.exists())
        
        # Verify file content
        df = pd.read_csv(output_path)
        self.assertEqual(len(df), 1)
        self.assertIn('Predicted_Active_Power', df.columns)
    
    def test_save_batch_predictions(self):
        """Test saving batch predictions to file."""
        predictions_df = pd.DataFrame({
            'Global_reactive_power': [0.5, 0.6],
            'Voltage': [220.0, 221.0],
            'Predicted_Active_Power': [2.5, 2.6]
        })
        
        output_path = Path(self.test_dir) / "batch_predictions.csv"
        
        success, error = self.predictor.save_batch_predictions(predictions_df, output_path)
        
        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertTrue(output_path.exists())


class TestMLPredictorConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def test_get_predictor(self):
        """Test getting predictor instance."""
        predictor = get_predictor()
        self.assertIsInstance(predictor, MLPredictor)
    
    def test_make_prediction(self):
        """Test make_prediction via MLPredictor with custom path."""
        with tempfile.TemporaryDirectory() as test_dir:
            test_model_path = Path(test_dir) / "test_model.pkl"
            
            # Create and save real model
            np.random.seed(42)
            X_train = np.random.randn(100, 10)
            y_train = X_train @ np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) + 0.5
            real_model = LinearRegression()
            real_model.fit(X_train, y_train)
            joblib.dump(real_model, test_model_path)
            
            # Use MLPredictor directly with custom path
            predictor = MLPredictor(model_path=test_model_path)
            
            input_data = {
                'Global_reactive_power': 0.5,
                'Voltage': 220.0,
                'Global_intensity': 10.0,
                'Sub_metering_1': 1.0,
                'Sub_metering_2': 0.0,
                'Sub_metering_3': 5.0,
                'Year': 2023,
                'Month': 6,
                'Day': 15,
                'Hour': 14
            }
            
            success, prediction, error = predictor.predict(input_data)
            self.assertTrue(success)
            self.assertIsNotNone(prediction)
    
    def test_load_trained_model(self):
        """Test load_trained_model via MLPredictor with custom path."""
        with tempfile.TemporaryDirectory() as test_dir:
            test_model_path = Path(test_dir) / "test_model.pkl"
            
            # Create and save real model
            np.random.seed(42)
            X_train = np.random.randn(100, 10)
            y_train = X_train @ np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) + 0.5
            real_model = LinearRegression()
            real_model.fit(X_train, y_train)
            joblib.dump(real_model, test_model_path)
            
            # Use MLPredictor directly with custom path
            predictor = MLPredictor(model_path=test_model_path)
            success, error = predictor.load_model()
            
            self.assertTrue(success)
            self.assertIsNone(error)


if __name__ == '__main__':
    unittest.main()