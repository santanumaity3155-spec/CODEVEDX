"""
Data handling functions for the Fake News Detection Tool.
Manages dataset loading, CSV operations, prediction history, and exports.
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd

from src.config import (
    PROCESSED_DATASET_PATH,
    PREDICTIONS_DIR,
    PREDICTION_HISTORY_FILE,
    CHARTS_DIR,
    REPORTS_DIR
)
from src.logger import logger


class DataHandler:
    """Handles all data operations for the application."""
    
    def __init__(self):
        """Initialize the DataHandler."""
        self.dataset_df: Optional[pd.DataFrame] = None
        self._load_dataset()
    
    def _load_dataset(self) -> bool:
        """
        Load the processed dataset.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if PROCESSED_DATASET_PATH.exists():
                self.dataset_df = pd.read_csv(PROCESSED_DATASET_PATH)
                logger.info(f"Dataset loaded: {len(self.dataset_df)} rows")
                return True
            else:
                logger.warning(f"Dataset not found at {PROCESSED_DATASET_PATH}")
                return False
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return False
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary with dataset information
        """
        if self.dataset_df is None:
            return {"error": "Dataset not loaded"}
        
        try:
            info = {
                "name": "Fake News Dataset",
                "rows": len(self.dataset_df),
                "columns": len(self.dataset_df.columns),
                "column_names": self.dataset_df.columns.tolist(),
                "class_distribution": self.dataset_df['label'].value_counts().to_dict() if 'label' in self.dataset_df.columns else {},
                "vocabulary_size": "20,000 (TF-IDF)",
                "location": str(PROCESSED_DATASET_PATH),
                "memory_usage_mb": round(self.dataset_df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
            return info
        except Exception as e:
            logger.error(f"Error getting dataset info: {str(e)}")
            return {"error": str(e)}
    
    def load_csv_for_batch(self, file_path: str) -> tuple[Optional[pd.DataFrame], str]:
        """
        Load a CSV file for batch prediction.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Tuple of (DataFrame or None, message)
        """
        try:
            df = pd.read_csv(file_path)
            message = f"Loaded {len(df)} rows with {len(df.columns)} columns"
            logger.info(f"Batch CSV loaded: {file_path} - {message}")
            return df, message
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return None, f"Error: {str(e)}"
    
    def save_predictions(self, predictions_df: pd.DataFrame, filename: str = "predictions.csv") -> tuple[bool, str]:
        """
        Save predictions to CSV file.
        
        Args:
            predictions_df: DataFrame with predictions
            filename: Output filename
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            output_path = PREDICTIONS_DIR / filename
            predictions_df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
            return True, f"Predictions saved to: {output_path}"
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
            return False, f"Error saving predictions: {str(e)}"
    
    def load_prediction_history(self) -> pd.DataFrame:
        """
        Load prediction history from file.
        
        Returns:
            DataFrame with prediction history
        """
        try:
            if PREDICTION_HISTORY_FILE.exists():
                df = pd.read_csv(PREDICTION_HISTORY_FILE)
                logger.info(f"Prediction history loaded: {len(df)} records")
                return df
            else:
                # Create empty DataFrame with expected columns
                df = pd.DataFrame(columns=[
                    'timestamp', 'input_length', 'prediction', 
                    'confidence', 'probability_fake', 'probability_real'
                ])
                return df
        except Exception as e:
            logger.error(f"Error loading prediction history: {str(e)}")
            return pd.DataFrame(columns=[
                'timestamp', 'input_length', 'prediction', 
                'confidence', 'probability_fake', 'probability_real'
            ])
    
    def save_prediction_history(self, history_df: pd.DataFrame) -> bool:
        """
        Save prediction history to file.
        
        Args:
            history_df: DataFrame with prediction history
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            history_df.to_csv(PREDICTION_HISTORY_FILE, index=False)
            logger.info(f"Prediction history saved: {len(history_df)} records")
            return True
        except Exception as e:
            logger.error(f"Error saving prediction history: {str(e)}")
            return False
    
    def add_prediction_to_history(self, prediction_data: Dict[str, Any]) -> bool:
        """
        Add a new prediction to the history.
        
        Args:
            prediction_data: Dictionary with prediction details
            
        Returns:
            True if added successfully, False otherwise
        """
        try:
            history_df = self.load_prediction_history()
            
            # Create new row
            new_row = pd.DataFrame([prediction_data])
            history_df = pd.concat([history_df, new_row], ignore_index=True)
            
            # Keep only last 10000 records to prevent file from growing too large
            if len(history_df) > 10000:
                history_df = history_df.tail(10000)
            
            return self.save_prediction_history(history_df)
        except Exception as e:
            logger.error(f"Error adding prediction to history: {str(e)}")
            return False
    
    def clear_prediction_history(self) -> bool:
        """
        Clear all prediction history.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            if PREDICTION_HISTORY_FILE.exists():
                PREDICTION_HISTORY_FILE.unlink()
            
            # Create empty file
            empty_df = pd.DataFrame(columns=[
                'timestamp', 'input_length', 'prediction', 
                'confidence', 'probability_fake', 'probability_real'
            ])
            empty_df.to_csv(PREDICTION_HISTORY_FILE, index=False)
            
            logger.info("Prediction history cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing prediction history: {str(e)}")
            return False
    
    def get_prediction_history(self, limit: int = 50) -> pd.DataFrame:
        """
        Get recent prediction history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with prediction history
        """
        try:
            history_df = self.load_prediction_history()
            if len(history_df) > limit:
                history_df = history_df.tail(limit)
            return history_df
        except Exception as e:
            logger.error(f"Error getting prediction history: {str(e)}")
            return pd.DataFrame()
    
    def export_predictions(self, history_df: pd.DataFrame, format_type: str, 
                          output_path: Optional[Path] = None) -> tuple[bool, str]:
        """
        Export predictions to different formats.
        
        Args:
            history_df: DataFrame with predictions
            format_type: Export format ('csv', 'json', 'txt')
            output_path: Optional custom output path
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if output_path is None:
                filename = f"predictions_export_{timestamp}.{format_type}"
                output_path = PREDICTIONS_DIR / filename
            
            format_type = format_type.lower()
            
            if format_type == 'csv':
                history_df.to_csv(output_path, index=False)
            elif format_type == 'json':
                history_df.to_json(output_path, orient='records', indent=2)
            elif format_type == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 70 + "\n")
                    f.write("FAKE NEWS DETECTION - PREDICTION EXPORT\n")
                    f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 70 + "\n\n")
                    
                    for idx, row in history_df.iterrows():
                        f.write(f"Prediction #{idx + 1}\n")
                        f.write("-" * 70 + "\n")
                        f.write(f"Timestamp: {row.get('timestamp', 'N/A')}\n")
                        f.write(f"Input Length: {row.get('input_length', 'N/A')} characters\n")
                        f.write(f"Prediction: {row.get('prediction', 'N/A')}\n")
                        f.write(f"Confidence: {row.get('confidence', 'N/A')}\n")
                        f.write(f"Probability (Fake): {row.get('probability_fake', 'N/A')}\n")
                        f.write(f"Probability (Real): {row.get('probability_real', 'N/A')}\n")
                        f.write("\n")
            else:
                return False, f"Unsupported format: {format_type}"
            
            logger.info(f"Predictions exported to {output_path}")
            return True, f"Export successful: {output_path}"
        
        except Exception as e:
            logger.error(f"Error exporting predictions: {str(e)}")
            return False, f"Error exporting predictions: {str(e)}"
    
    def get_model_files_info(self) -> Dict[str, Any]:
        """
        Get information about model files.
        
        Returns:
            Dictionary with model file information
        """
        from src.config import MODEL_PATH, VECTORIZER_PATH
        
        info = {
            "model": {
                "path": str(MODEL_PATH),
                "exists": MODEL_PATH.exists(),
                "size": get_file_size(MODEL_PATH) if MODEL_PATH.exists() else "N/A"
            },
            "vectorizer": {
                "path": str(VECTORIZER_PATH),
                "exists": VECTORIZER_PATH.exists(),
                "size": get_file_size(VECTORIZER_PATH) if VECTORIZER_PATH.exists() else "N/A"
            }
        }
        return info


def get_file_size(file_path: Path) -> str:
    """
    Get formatted file size.
    
    Args:
        file_path: Path to file
        
    Returns:
        Formatted file size string
    """
    try:
        if file_path.exists():
            size = file_path.stat().st_size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.2f} {unit}"
                size /= 1024.0
            return f"{size:.2f} TB"
        return "N/A"
    except Exception:
        return "N/A"


# Create a global instance
data_handler = DataHandler()