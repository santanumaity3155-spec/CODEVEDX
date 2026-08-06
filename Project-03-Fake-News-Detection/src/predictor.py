"""
Prediction engine for the Fake News Detection Tool.
Handles model loading, text preprocessing, and prediction generation.
"""

import re
import joblib
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np

from src.config import MODEL_PATH, VECTORIZER_PATH, MODEL_INFO
from src.logger import logger


class NewsPredictor:
    """Handles news prediction using the trained model."""
    
    def __init__(self):
        """Initialize the predictor by loading model and vectorizer."""
        self.model = None
        self.vectorizer = None
        self.model_loaded = False
        self.vectorizer_loaded = False
        self.load_error = None
        
        # Load model and vectorizer
        self._load_model()
        self._load_vectorizer()
    
    def _load_model(self) -> bool:
        """
        Load the trained model from file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not MODEL_PATH.exists():
                self.load_error = f"Model file not found: {MODEL_PATH}"
                logger.error(self.load_error)
                return False
            
            self.model = joblib.load(MODEL_PATH)
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
            return True
            
        except Exception as e:
            self.load_error = f"Error loading model: {str(e)}"
            logger.error(self.load_error)
            return False
    
    def _load_vectorizer(self) -> bool:
        """
        Load the TF-IDF vectorizer from file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not VECTORIZER_PATH.exists():
                self.load_error = f"Vectorizer file not found: {VECTORIZER_PATH}"
                logger.error(self.load_error)
                return False
            
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            
            self.vectorizer_loaded = True
            logger.info(f"Vectorizer loaded successfully from {VECTORIZER_PATH}")
            return True
            
        except Exception as e:
            self.load_error = f"Error loading vectorizer: {str(e)}"
            logger.error(self.load_error)
            return False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for prediction.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short words (optional, helps with noise reduction)
        text = ' '.join([word for word in text.split() if len(word) > 2])
        
        return text
    
    def is_ready(self) -> bool:
        """
        Check if the predictor is ready to make predictions.
        
        Returns:
            True if both model and vectorizer are loaded, False otherwise
        """
        return self.model_loaded and self.vectorizer_loaded
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the predictor.
        
        Returns:
            Dictionary with status information
        """
        return {
            "model_loaded": self.model_loaded,
            "vectorizer_loaded": self.vectorizer_loaded,
            "ready": self.is_ready(),
            "load_error": self.load_error,
            "model_path": str(MODEL_PATH),
            "vectorizer_path": str(VECTORIZER_PATH)
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Make a prediction for the given text.
        
        Args:
            text: News text to classify
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_ready():
            error_msg = self.load_error or "Model or vectorizer not loaded"
            logger.error(f"Prediction failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "prediction": None,
                "confidence": 0.0,
                "probability_fake": 0.0,
                "probability_real": 0.0,
                "processing_time": 0.0,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            start_time = time.time()
            
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            if not cleaned_text:
                return {
                    "success": False,
                    "error": "Text is empty after cleaning",
                    "prediction": None,
                    "confidence": 0.0,
                    "probability_fake": 0.0,
                    "probability_real": 0.0,
                    "processing_time": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Vectorize the text
            text_vectorized = self.vectorizer.transform([cleaned_text])
            
            # Make prediction
            prediction = self.model.predict(text_vectorized)[0]
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(text_vectorized)[0]
                prob_fake = float(probabilities[0])
                prob_real = float(probabilities[1])
                confidence = max(prob_fake, prob_real)
            else:
                # If no probability method, use decision function or binary prediction
                prob_fake = 0.5
                prob_real = 0.5
                confidence = 0.5
                logger.warning("Model does not support probability predictions")
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Format result
            result = {
                "success": True,
                "error": None,
                "prediction": "FAKE NEWS" if prediction == 0 else "REAL NEWS",
                "prediction_label": int(prediction),
                "confidence": round(confidence, 4),
                "probability_fake": round(prob_fake, 4),
                "probability_real": round(prob_real, 4),
                "processing_time": round(processing_time, 4),
                "input_length": len(text),
                "cleaned_length": len(cleaned_text),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Prediction made: {result['prediction']} (confidence: {confidence:.2%})")
            return result
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time if 'start_time' in locals() else 0.0
            
            error_msg = f"Error during prediction: {str(e)}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "prediction": None,
                "confidence": 0.0,
                "probability_fake": 0.0,
                "probability_real": 0.0,
                "processing_time": round(processing_time, 4),
                "timestamp": datetime.now().isoformat()
            }
    
    def predict_batch(self, texts: list) -> list:
        """
        Make predictions for multiple texts.
        
        Args:
            texts: List of news texts to classify
            
        Returns:
            List of prediction result dictionaries
        """
        results = []
        total = len(texts)
        
        logger.info(f"Starting batch prediction for {total} texts")
        
        for idx, text in enumerate(texts, 1):
            if idx % 10 == 0 or idx == total:
                logger.info(f"Batch prediction progress: {idx}/{total}")
            
            result = self.predict(text)
            result['batch_index'] = idx
            results.append(result)
        
        successful = sum(1 for r in results if r['success'])
        logger.info(f"Batch prediction completed: {successful}/{total} successful")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = MODEL_INFO.copy()
        
        # Add runtime information
        info["model_loaded"] = self.model_loaded
        info["vectorizer_loaded"] = self.vectorizer_loaded
        info["ready"] = self.is_ready()
        
        # Add file sizes
        if MODEL_PATH.exists():
            info["model_file_size"] = self._get_file_size(MODEL_PATH)
        if VECTORIZER_PATH.exists():
            info["vectorizer_file_size"] = self._get_file_size(VECTORIZER_PATH)
        
        # Add model details if loaded
        if self.model is not None:
            info["model_type"] = type(self.model).__name__
            if hasattr(self.model, 'n_estimators'):
                info["n_estimators"] = self.model.n_estimators
            if hasattr(self.model, 'max_depth'):
                info["max_depth"] = self.model.max_depth
        
        if self.vectorizer is not None:
            info["vocabulary_size"] = len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else "N/A"
        
        return info
    
    def _get_file_size(self, file_path: Path) -> str:
        """
        Get formatted file size.
        
        Args:
            file_path: Path to file
            
        Returns:
            Formatted file size string
        """
        try:
            size = file_path.stat().st_size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.2f} {unit}"
                size /= 1024.0
            return f"{size:.2f} TB"
        except Exception:
            return "N/A"


# Create a global instance
predictor = NewsPredictor()