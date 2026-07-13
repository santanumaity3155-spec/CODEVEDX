"""
ML Predictor Module for Utility Usage Prediction Tool

This module provides machine learning prediction functionality including:
- Loading trained models
- Making predictions on new data
- Displaying prediction results
- Saving predictions to files

Author: CodeVedX AI/ML Internship
"""

import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from config import (
    MODEL_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    PREDICTIONS_DIR
)
from logger import get_logger, log_prediction_made, log_error_occurred
from utils import print_success, print_error, print_info, format_number


# ========================================
# ML PREDICTOR CLASS
# ========================================

class MLPredictor:
    """
    Machine Learning predictor class for utility usage prediction.
    
    This class provides methods to load trained models and make predictions
    on new data using the trained Linear Regression model.
    """
    
    def __init__(self, model_path: Path = MODEL_PATH):
        """
        Initialize the ML predictor.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.logger = get_logger("MLPredictor")
        self.is_loaded = False
    
    # ========================================
    # MODEL MANAGEMENT
    # ========================================
    
    def load_model(self) -> Tuple[bool, Optional[str]]:
        """
        Load the trained model from file.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Check if model file exists
            if not self.model_path.exists():
                error_msg = f"Model file not found at {self.model_path}. Please train the model first."
                self.logger.error(error_msg)
                return False, error_msg
            
            # Load the model
            self.model = joblib.load(self.model_path)
            self.is_loaded = True
            
            self.logger.info(f"Model loaded successfully from {self.model_path}")
            print_success(f"Model loaded successfully from {self.model_path.name}")
            
            return True, None
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.is_loaded = False
            return False, error_msg
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.is_loaded and self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if not self.is_model_loaded():
            return {"error": "Model not loaded"}
        
        try:
            info = {
                "model_type": "Linear Regression",
                "model_path": str(self.model_path),
                "file_exists": self.model_path.exists(),
                "file_size": f"{self.model_path.stat().st_size / 1024**2:.2f} MB",
                "features_count": len(FEATURE_COLUMNS),
                "features": FEATURE_COLUMNS,
                "target": TARGET_COLUMN,
                "is_loaded": self.is_loaded
            }
            
            # Add model coefficients if available
            if hasattr(self.model, 'coef_'):
                info["coefficients"] = self.model.coef_.tolist()
                info["intercept"] = float(self.model.intercept_)
            
            return info
            
        except Exception as e:
            return {"error": str(e)}
    
    # ========================================
    # PREDICTION METHODS
    # ========================================
    
    def predict(self, input_data: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[str]]:
        """
        Make a prediction using the loaded model.
        
        Args:
            input_data: Dictionary containing input features
        
        Returns:
            Tuple of (success, predicted_value, error_message)
        """
        try:
            # Check if model is loaded
            if not self.is_model_loaded():
                success, error = self.load_model()
                if not success:
                    return False, None, error
            
            # Prepare input dataframe
            input_df = self._prepare_input(input_data)
            if input_df is None:
                return False, None, "Failed to prepare input data"
            
            # Make prediction
            prediction = self.model.predict(input_df)[0]
            
            # Log the prediction
            log_prediction_made(input_data, prediction)
            
            self.logger.info(f"Prediction made: {prediction:.4f} kW")
            
            return True, float(prediction), None
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            log_error_occurred(type(e).__name__, str(e))
            return False, None, error_msg
    
    def predict_batch(self, input_data: pd.DataFrame) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Make predictions for multiple records.
        
        Args:
            input_data: DataFrame containing input features
        
        Returns:
            Tuple of (success, predictions_df, error_message)
        """
        try:
            # Check if model is loaded
            if not self.is_model_loaded():
                success, error = self.load_model()
                if not success:
                    return False, None, error
            
            # Select only feature columns
            input_df = input_data[FEATURE_COLUMNS]
            
            # Make predictions
            predictions = self.model.predict(input_df)
            
            # Create results dataframe
            results_df = input_data.copy()
            results_df['Predicted_Active_Power'] = predictions
            
            self.logger.info(f"Batch prediction completed: {len(predictions)} predictions")
            
            return True, results_df, None
            
        except Exception as e:
            error_msg = f"Batch prediction failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, None, error_msg
    
    def _prepare_input(self, input_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Prepare input data for prediction.
        
        Args:
            input_data: Dictionary containing input features
        
        Returns:
            DataFrame with prepared input or None if failed
        """
        try:
            # Create input dataframe with single row
            input_df = pd.DataFrame([input_data])
            
            # Ensure all feature columns are present
            for col in FEATURE_COLUMNS:
                if col not in input_df.columns:
                    self.logger.error(f"Missing feature: {col}")
                    return None
            
            # Select only feature columns in correct order
            input_df = input_df[FEATURE_COLUMNS]
            
            # Convert to numeric
            for col in FEATURE_COLUMNS:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            
            # Check for NaN values
            if input_df.isnull().any().any():
                self.logger.error("Input data contains NaN values")
                return None
            
            return input_df
            
        except Exception as e:
            self.logger.error(f"Failed to prepare input: {str(e)}", exc_info=True)
            return None
    
    # ========================================
    # DISPLAY METHODS
    # ========================================
    
    def display_prediction(self, prediction: float, input_data: Dict[str, Any]) -> None:
        """
        Display prediction results in a formatted way.
        
        Args:
            prediction: Predicted value
            input_data: Input features used for prediction
        """
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        
        print(f"\n{'Input Features':^60}")
        print("-" * 60)
        for key, value in input_data.items():
            if isinstance(value, float):
                value = format_number(value, 4)
            print(f"{key:30s}: {value}")
        
        print(f"\n{'Prediction':^60}")
        print("-" * 60)
        print(f"{'Predicted Active Power':30s}: {format_number(prediction, 4)} kW")
        print("=" * 60)
    
    def display_model_metrics(self) -> None:
        """Display model information and metrics."""
        if not self.is_model_loaded():
            print_error("Model not loaded. Please load or train the model first.")
            return
        
        info = self.get_model_info()
        
        print("\n" + "=" * 60)
        print("MODEL INFORMATION")
        print("=" * 60)
        
        print(f"\n{'Model Type':30s}: {info.get('model_type', 'N/A')}")
        print(f"{'Model Path':30s}: {info.get('model_path', 'N/A')}")
        print(f"{'File Size':30s}: {info.get('file_size', 'N/A')}")
        print(f"{'Features Count':30s}: {info.get('features_count', 'N/A')}")
        print(f"{'Target Variable':30s}: {info.get('target', 'N/A')}")
        
        if 'coefficients' in info:
            print(f"\n{'Model Coefficients':^60}")
            print("-" * 60)
            for feature, coef in zip(info['features'], info['coefficients']):
                print(f"{feature:30s}: {coef:+.6f}")
            print(f"{'Intercept':30s}: {info['intercept']:+.6f}")
        
        print("=" * 60)
    
    # ========================================
    # PREDICTION STORAGE
    # ========================================
    
    def save_prediction(
        self,
        prediction: float,
        input_data: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Save prediction to CSV file.
        
        Args:
            prediction: Predicted value
            input_data: Input features used for prediction
            output_path: Path to save prediction (None for default)
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Use default path if not provided
            if output_path is None:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                output_path = PREDICTIONS_DIR / f"prediction_{timestamp}.csv"
            
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create prediction record
            record = input_data.copy()
            record['Predicted_Active_Power'] = prediction
            record['Prediction_Timestamp'] = pd.Timestamp.now().isoformat()
            
            # Create dataframe
            df = pd.DataFrame([record])
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Prediction saved to: {output_path}")
            print_success(f"Prediction saved to: {output_path}")
            
            return True, None
            
        except Exception as e:
            error_msg = f"Failed to save prediction: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def save_batch_predictions(
        self,
        predictions_df: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Save batch predictions to CSV file.
        
        Args:
            predictions_df: DataFrame containing predictions
            output_path: Path to save predictions (None for default)
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Use default path if not provided
            if output_path is None:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                output_path = PREDICTIONS_DIR / f"batch_predictions_{timestamp}.csv"
            
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            predictions_df.to_csv(output_path, index=False)
            
            self.logger.info(f"Batch predictions saved to: {output_path}")
            print_success(f"Batch predictions saved to: {output_path}")
            
            return True, None
            
        except Exception as e:
            error_msg = f"Failed to save batch predictions: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    # ========================================
    # VALIDATION METHODS
    # ========================================
    
    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate input data for prediction.
        
        Args:
            input_data: Dictionary containing input features
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if all required features are present
        missing_features = [col for col in FEATURE_COLUMNS if col not in input_data]
        if missing_features:
            return False, f"Missing features: {', '.join(missing_features)}"
        
        # Check for None values
        none_values = [col for col in FEATURE_COLUMNS if input_data.get(col) is None]
        if none_values:
            return False, f"Null values in features: {', '.join(none_values)}"
        
        # Check for valid numeric values
        for col in FEATURE_COLUMNS:
            try:
                float(input_data[col])
            except (ValueError, TypeError):
                return False, f"Invalid numeric value for {col}: {input_data[col]}"
        
        return True, None
    
    # ========================================
    # UTILITY METHODS
    # ========================================
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from the model.
        
        Returns:
            Dictionary mapping features to their coefficients or None
        """
        if not self.is_model_loaded() or not hasattr(self.model, 'coef_'):
            return None
        
        importance = {}
        for feature, coef in zip(FEATURE_COLUMNS, self.model.coef_):
            importance[feature] = float(coef)
        
        return importance
    
    def explain_prediction(self, input_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate explanation for a prediction.
        
        Args:
            input_data: Input features used for prediction
        
        Returns:
            Explanation string or None
        """
        if not self.is_model_loaded():
            return None
        
        try:
            # Get feature importance
            importance = self.get_feature_importance()
            if not importance:
                return None
            
            # Sort features by absolute importance
            sorted_features = sorted(
                importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Generate explanation
            explanation = "Prediction Explanation:\n"
            explanation += "-" * 60 + "\n"
            explanation += "Top contributing features:\n"
            
            for feature, coef in sorted_features[:5]:
                impact = "positive" if coef > 0 else "negative"
                explanation += f"  • {feature:25s}: {coef:+.6f} ({impact} impact)\n"
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Failed to explain prediction: {str(e)}")
            return None


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================


def get_predictor() -> MLPredictor:
    """
    Get an MLPredictor instance.
    
    Returns:
        MLPredictor instance
    """
    return MLPredictor()


def make_prediction(input_data: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Make a prediction (convenience function).
    
    Args:
        input_data: Dictionary containing input features
    
    Returns:
        Tuple of (success, predicted_value, error_message)
    """
    predictor = get_predictor()
    return predictor.predict(input_data)


def load_trained_model() -> Tuple[bool, Optional[str]]:
    """
    Load the trained model (convenience function).
    
    Returns:
        Tuple of (success, error_message)
    """
    predictor = get_predictor()
    return predictor.load_model()