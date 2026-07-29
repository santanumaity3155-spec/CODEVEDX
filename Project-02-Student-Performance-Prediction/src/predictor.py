"""
Predictor Module
Handles model loading, single and batch predictions, and prediction management.
"""

import json
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from .config import MODEL_PATH, PREDICTIONS_DIR, MODEL_FEATURES
from .logger import app_logger, log_prediction, log_error, log_warning
from .utils import print_success, print_error, print_info, print_header, export_to_csv, get_filename_timestamp
from .validation import validate_prediction_input


class ModelLoadError(Exception):
    """Exception raised when model fails to load."""
    pass


class PredictionError(Exception):
    """Exception raised when prediction fails."""
    pass


class Predictor:
    """
    Handles model loading and prediction operations.
    """
    
    def __init__(self, model_path: Path = MODEL_PATH):
        """
        Initialize Predictor.
        
        Args:
            model_path: Path to the trained model pickle file
        """
        self.model_path = model_path
        self.model = None
        self.model_info: Dict[str, Any] = {}
        self.is_loaded = False
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load the trained model from pickle file.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not self.model_path.exists():
                error_msg = f"Model file not found: {self.model_path}"
                log_error(error_msg)
                print_error(error_msg)
                print_info("Please ensure the model has been trained and saved.")
                return False
            
            app_logger.info(f"Loading model from: {self.model_path}")
            
            # Load model using joblib
            model_data = joblib.load(self.model_path)
            
            # Handle different model storage formats
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.model_info = model_data.get('info', {})
            else:
                self.model = model_data
                self.model_info = {}
            
            if self.model is None:
                raise ModelLoadError("Model data is None")
            
            self.is_loaded = True
            app_logger.info("Model loaded successfully")
            print_success("Model loaded successfully")
            
            # Log model information
            if self.model_info:
                app_logger.info(f"Model info: {self.model_info}")
            
            return True
        
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            log_error(error_msg, e)
            print_error(error_msg)
            self.is_loaded = False
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if not self.is_loaded or self.model is None:
            return {'status': 'Not Loaded'}
        
        info = {
            'status': 'Loaded',
            'model_path': str(self.model_path),
            'model_type': type(self.model).__name__,
        }
        
        # Add model-specific information
        if hasattr(self.model, 'feature_importances_'):
            info['has_feature_importances'] = True
        
        if hasattr(self.model, 'n_features_in_'):
            info['n_features'] = self.model.n_features_in_
        
        # Add stored model info
        info.update(self.model_info)
        
        return info
    
    def print_model_info(self) -> None:
        """
        Print formatted model information.
        """
        from .utils import print_header, print_subheader, print_separator
        
        print_header("MODEL INFORMATION")
        
        info = self.get_model_info()
        
        print(f"\nStatus: {info.get('status', 'Unknown')}")
        print(f"Model Name: student_performance_model.pkl")
        print(f"Algorithm: {info.get('model_type', 'N/A')}")
        print(f"Model Path: {info.get('model_path', 'N/A')}")
        
        # File information
        if self.model_path.exists():
            file_stat = self.model_path.stat()
            file_size = file_stat.st_size / (1024 * 1024)  # Convert to MB
            last_modified = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"File Size: {file_size:.2f} MB")
            print(f"Last Modified: {last_modified}")
        
        # Training date if available
        if 'training_date' in info:
            print(f"Training Date: {info['training_date']}")
        elif 'created_at' in info:
            print(f"Training Date: {info['created_at']}")
        
        # Number of features
        if 'n_features' in info:
            print(f"Number of Features: {info['n_features']}")
        elif hasattr(self.model, 'n_features_in_'):
            print(f"Number of Features: {self.model.n_features_in_}")
        
        # Model loaded status
        print(f"Model Loaded Successfully: {'Yes' if self.is_loaded else 'No'}")
        
        # Feature importances
        if self.is_loaded and hasattr(self.model, 'feature_importances_'):
            print_subheader("FEATURE IMPORTANCES")
            
            feature_names = MODEL_FEATURES
            importances = self.model.feature_importances_
            
            # Sort by importance
            sorted_indices = np.argsort(importances)[::-1]
            
            for i, idx in enumerate(sorted_indices[:10]):  # Show top 10
                if i < len(feature_names):
                    feature_name = feature_names[idx]
                    importance = importances[idx]
                    print(f"{i+1:2d}. {feature_name:20s}: {importance:.4f}")
        
        # Additional information
        if self.model_info:
            print_subheader("ADDITIONAL INFORMATION")
            for key, value in self.model_info.items():
                # Skip keys already displayed
                if key not in ['training_date', 'created_at', 'n_features']:
                    print(f"{key}: {value}")
        
        print_separator()
    
    def prepare_input_data(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare input data for prediction.
        
        Args:
            input_data: Dictionary containing input features
        
        Returns:
            DataFrame ready for prediction
        """
        # Extract only the features needed by the model
        features = {}
        for feature in MODEL_FEATURES:
            if feature in input_data:
                features[feature] = [input_data[feature]]
            else:
                raise PredictionError(f"Missing required feature: {feature}")
        
        # Create DataFrame
        df = pd.DataFrame(features)
        
        return df
    
    def predict_single(self, input_data: Dict[str, Any]) -> Tuple[bool, Optional[float], str]:
        """
        Make a single prediction.
        
        Args:
            input_data: Dictionary containing input features
        
        Returns:
            Tuple of (success, predicted_score, message)
        """
        try:
            if not self.is_loaded or self.model is None:
                error_msg = f"Model not loaded. Cannot make predictions. Expected model at: {self.model_path}"
                log_error(error_msg)
                return False, None, error_msg
            
            # Validate input
            is_valid, errors = validate_prediction_input(input_data)
            if not is_valid:
                error_msg = "; ".join(errors)
                log_warning(f"Prediction input validation failed: {error_msg}")
                return False, None, f"Validation failed: {error_msg}"
            
            # Prepare data
            df = self.prepare_input_data(input_data)
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            
            # Ensure prediction is within valid range
            prediction = max(0, min(100, prediction))
            
            # Log prediction
            log_prediction(input_data, prediction, success=True)
            
            return True, prediction, "Prediction successful"
        
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            log_error(error_msg, e)
            log_prediction(input_data, 0.0, success=False)
            return False, None, error_msg
    
    def predict_batch(self, input_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make batch predictions.
        
        Args:
            input_data_list: List of dictionaries containing input features
        
        Returns:
            List of prediction results
        """
        results = []
        
        for i, input_data in enumerate(input_data_list):
            try:
                success, prediction, message = self.predict_single(input_data)
                
                result = {
                    'index': i + 1,
                    'input': input_data,
                    'prediction': prediction if success else None,
                    'success': success,
                    'message': message
                }
                
                results.append(result)
            
            except Exception as e:
                log_error(f"Batch prediction failed for item {i+1}", e)
                results.append({
                    'index': i + 1,
                    'input': input_data,
                    'prediction': None,
                    'success': False,
                    'message': str(e)
                })
        
        return results
    
    def predict_from_csv(self, csv_path: Path) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Make predictions from a CSV file.
        
        Args:
            csv_path: Path to CSV file with input data
        
        Returns:
            Tuple of (success, results_list, message)
        """
        try:
            if not csv_path.exists():
                return False, [], f"CSV file not found: {csv_path}"
            
            # Read CSV
            from .utils import read_csv_file
            records, headers = read_csv_file(csv_path)
            
            if not records:
                return False, [], "CSV file is empty"
            
            # Make predictions
            results = self.predict_batch(records)
            
            # Add original data to results
            for i, result in enumerate(results):
                if i < len(records):
                    result['original_data'] = records[i]
            
            successful_predictions = sum(1 for r in results if r['success'])
            app_logger.info(f"Batch prediction completed: {successful_predictions}/{len(results)} successful")
            
            return True, results, f"Predicted {successful_predictions}/{len(results)} records"
        
        except Exception as e:
            error_msg = f"Batch prediction from CSV failed: {str(e)}"
            log_error(error_msg, e)
            return False, [], error_msg
    
    def save_predictions(
        self,
        predictions: List[Dict[str, Any]],
        output_path: Optional[Path] = None,
        filename_prefix: str = "predictions"
    ) -> Tuple[bool, Path]:
        """
        Save predictions to CSV file.
        
        Args:
            predictions: List of prediction results
            output_path: Custom output path (generates timestamped file if None)
            filename_prefix: Prefix for generated filename
        
        Returns:
            Tuple of (success, file_path)
        """
        try:
            # Ensure predictions directory exists
            PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Generate output path if not provided
            if output_path is None:
                timestamp = get_filename_timestamp()
                output_path = PREDICTIONS_DIR / f"{filename_prefix}_{timestamp}.csv"
            
            # Prepare data for export
            export_data = []
            
            for pred in predictions:
                if pred['success'] and pred['prediction'] is not None:
                    row = {
                        'index': pred['index'],
                        'predicted_score': round(pred['prediction'], 2),
                        'prediction_success': pred['success']
                    }
                    
                    # Add input features
                    if 'input' in pred:
                        for key, value in pred['input'].items():
                            row[key] = value
                    
                    # Add original data if available
                    if 'original_data' in pred:
                        for key, value in pred['original_data'].items():
                            if key not in row:
                                row[key] = value
                    
                    export_data.append(row)
            
            if not export_data:
                log_warning("No successful predictions to save")
                return False, output_path
            
            # Export to CSV
            success = export_to_csv(export_data, output_path)
            
            if success:
                app_logger.info(f"Predictions saved to: {output_path}")
                print_success(f"Predictions saved to: {output_path}")
                return True, output_path
            else:
                return False, output_path
        
        except Exception as e:
            error_msg = f"Failed to save predictions: {str(e)}"
            log_error(error_msg, e)
            return False, Path("")
    
    def display_prediction_result(self, prediction: float, input_data: Dict[str, Any]) -> None:
        """
        Display formatted prediction result.
        
        Args:
            prediction: Predicted score
            input_data: Input features used for prediction
        """
        from .utils import format_score, print_header, print_separator
        
        print_header("PREDICTION RESULT")
        
        print("\nInput Features:")
        print("-" * 80)
        
        # Display key input features
        key_features = [
            'Hours_Studied', 'Attendance', 'Sleep_Hours',
            'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity'
        ]
        
        for feature in key_features:
            if feature in input_data:
                print(f"{feature:20s}: {input_data[feature]}")
        
        print("\n" + "-" * 80)
        print(f"\n{'PREDICTED EXAM SCORE':^80}")
        print(f"\n{format_score(prediction):^80}")
        print("\n" + "-" * 80)
        
        # Provide interpretation
        percentage = (prediction / 100) * 100
        
        if percentage >= 90:
            interpretation = "Excellent! The student is likely to perform exceptionally well."
        elif percentage >= 80:
            interpretation = "Very Good! The student is likely to perform well."
        elif percentage >= 70:
            interpretation = "Good! The student is likely to have a satisfactory performance."
        elif percentage >= 60:
            interpretation = "Average. The student may need some additional support."
        elif percentage >= 50:
            interpretation = "Below Average. The student needs improvement and support."
        else:
            interpretation = "Poor. The student needs significant intervention and support."
        
        print(f"\nInterpretation: {interpretation}")
        print_separator()
    
    def display_batch_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Display formatted batch prediction results.
        
        Args:
            results: List of prediction results
        """
        from .utils import print_header, print_subheader, print_table, print_separator
        
        print_header("BATCH PREDICTION RESULTS")
        
        if not results:
            print_info("No results to display")
            return
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\nTotal Predictions: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            print_subheader("SUCCESSFUL PREDICTIONS")
            
            # Prepare table data
            headers = ['Index', 'Hours_Studied', 'Attendance', 'Previous_Scores', 'Predicted_Score']
            rows = []
            
            for result in successful[:20]:  # Show first 20
                if 'input' in result:
                    input_data = result['input']
                    row = [
                        str(result['index']),
                        str(input_data.get('Hours_Studied', 'N/A')),
                        str(input_data.get('Attendance', 'N/A')),
                        str(input_data.get('Previous_Scores', 'N/A')),
                        f"{result['prediction']:.2f}"
                    ]
                    rows.append(row)
            
            if rows:
                print_table(headers, rows)
            
            if len(successful) > 20:
                print(f"\n... and {len(successful) - 20} more results")
        
        if failed:
            print_subheader("FAILED PREDICTIONS")
            for result in failed[:10]:  # Show first 10
                print(f"Index {result['index']}: {result['message']}")
            
            if len(failed) > 10:
                print(f"\n... and {len(failed) - 10} more failures")
        
        print_separator()
    
    def reload_model(self) -> bool:
        """
        Reload the model from file.
        
        Returns:
            True if reloaded successfully, False otherwise
        """
        app_logger.info("Reloading model...")
        self.is_loaded = False
        self.model = None
        self.model_info = {}
        return self._load_model()
    
    def is_model_ready(self) -> bool:
        """
        Check if model is ready for predictions.
        
        Returns:
            True if model is loaded and ready, False otherwise
        """
        return self.is_loaded and self.model is not None
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.is_loaded