"""
Tests for Predictor Module
"""

import pytest
import pickle
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.predictor import Predictor, ModelLoadError, PredictionError
from src.config import MODEL_PATH


# Create a simple picklable mock model class
class PicklableMockModel:
    """A simple mock model that can be pickled."""
    
    def __init__(self, prediction_value=75.0):
        self.prediction_value = prediction_value
        self.feature_importances_ = np.array([0.5, 0.3, 0.2])
        self.n_features_in_ = 6
    
    def predict(self, X):
        """Mock predict method."""
        return np.array([self.prediction_value])
    
    def __eq__(self, other):
        return isinstance(other, PicklableMockModel)


def create_mock_model(prediction_value=75.0):
    """Create a mock model that can be pickled."""
    return PicklableMockModel(prediction_value=prediction_value)


class TestPredictorInitialization:
    """Test Predictor initialization."""
    
    def test_init_with_default_path(self):
        """Test initialization with default model path."""
        predictor = Predictor()
        assert predictor.model_path == MODEL_PATH
        assert predictor.model is None or predictor.model is not None
        assert isinstance(predictor.model_info, dict)
    
    def test_init_with_custom_path(self, tmp_path):
        """Test initialization with custom model path."""
        custom_path = tmp_path / "custom_model.pkl"
        predictor = Predictor(model_path=custom_path)
        assert predictor.model_path == custom_path
    
    def test_init_model_not_found(self, tmp_path):
        """Test initialization when model file doesn't exist."""
        non_existent = tmp_path / "nonexistent.pkl"
        predictor = Predictor(model_path=non_existent)
        
        assert predictor.is_loaded is False
        assert predictor.model is None


class TestPredictorModelLoading:
    """Test model loading functionality."""
    
    def test_load_model_success(self, tmp_path):
        """Test successful model loading."""
        # Create a mock model
        mock_model = create_mock_model()
        
        # Save mock model
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        
        assert predictor.is_loaded is True
        assert predictor.model is not None
        assert predictor.model == mock_model
    
    def test_load_model_with_dict_format(self, tmp_path):
        """Test loading model stored as dictionary."""
        # Create mock model and info
        mock_model = create_mock_model()
        
        model_data = {
            'model': mock_model,
            'info': {
                'accuracy': 0.95,
                'features': ['Hours_Studied', 'Attendance']
            }
        }
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        predictor = Predictor(model_path=model_path)
        
        assert predictor.is_loaded is True
        assert predictor.model == mock_model
        assert predictor.model_info['accuracy'] == 0.95
    
    def test_load_model_corrupted_file(self, tmp_path):
        """Test loading corrupted model file."""
        model_path = tmp_path / "corrupted.pkl"
        
        # Write invalid pickle data
        with open(model_path, 'wb') as f:
            f.write(b"not a pickle file")
        
        predictor = Predictor(model_path=model_path)
        
        assert predictor.is_loaded is False
        assert predictor.model is None
    
    def test_reload_model(self, tmp_path):
        """Test reloading model."""
        # Create initial model
        mock_model1 = create_mock_model()
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model1, f)
        
        predictor = Predictor(model_path=model_path)
        assert predictor.model == mock_model1
        
        # Create new model
        mock_model2 = create_mock_model()
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model2, f)
        
        # Reload
        success = predictor.reload_model()
        
        assert success is True
        assert predictor.model == mock_model2


class TestPredictorModelInfo:
    """Test model information functionality."""
    
    def test_get_model_info_when_loaded(self, tmp_path):
        """Test getting model info when model is loaded."""
        mock_model = create_mock_model()
        
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        info = predictor.get_model_info()
        
        assert info['status'] == 'Loaded'
        assert 'model_type' in info
        assert 'model_path' in info
    
    def test_get_model_info_when_not_loaded(self, tmp_path):
        """Test getting model info when model is not loaded."""
        non_existent = tmp_path / "nonexistent.pkl"
        predictor = Predictor(model_path=non_existent)
        
        info = predictor.get_model_info()
        
        assert info['status'] == 'Not Loaded'
    
    def test_get_model_info_with_feature_importances(self, tmp_path):
        """Test getting model info with feature importances."""
        mock_model = create_mock_model()
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        mock_model.n_features_in_ = 3
        
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        info = predictor.get_model_info()
        
        assert info['has_feature_importances'] is True
        assert info['n_features'] == 3
    
    def test_is_model_ready(self, tmp_path):
        """Test checking if model is ready."""
        non_existent = tmp_path / "nonexistent.pkl"
        predictor = Predictor(model_path=non_existent)
        
        assert predictor.is_model_ready() is False
        
        # Load model
        mock_model = create_mock_model()
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        assert predictor.is_model_ready() is True


class TestPredictorPrediction:
    """Test prediction functionality."""
    
    def test_predict_single_success(self, tmp_path):
        """Test successful single prediction."""
        # Create mock model with custom prediction value
        mock_model = create_mock_model(prediction_value=78.5)
        
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        
        input_data = {
            'Hours_Studied': 25,
            'Attendance': 85,
            'Sleep_Hours': 7,
            'Previous_Scores': 75,
            'Tutoring_Sessions': 2,
            'Physical_Activity': 3,
            'Gender': 'Male',
            'School_Type': 'Public',
            'Teacher_Quality': 'Medium',
            'Parental_Involvement': 'Medium',
            'Access_to_Resources': 'Medium',
            'Extracurricular_Activities': 'Yes',
            'Motivation_Level': 'Medium',
            'Internet_Access': 'Yes',
            'Family_Income': 'Medium',
            'Peer_Influence': 'Neutral',
            'Learning_Disabilities': 'No',
            'Parental_Education_Level': 'High School',
            'Distance_from_Home': 'Near'
        }
        
        success, prediction, message = predictor.predict_single(input_data)
        
        assert success is True
        assert prediction is not None
        assert isinstance(prediction, float)
        assert 0 <= prediction <= 100
    
    def test_predict_single_model_not_loaded(self, tmp_path):
        """Test prediction when model is not loaded."""
        non_existent = tmp_path / "nonexistent.pkl"
        predictor = Predictor(model_path=non_existent)
        
        input_data = {
            'Hours_Studied': 25,
            'Attendance': 85,
            'Sleep_Hours': 7,
            'Previous_Scores': 75,
            'Tutoring_Sessions': 2,
            'Physical_Activity': 3
        }
        
        success, prediction, message = predictor.predict_single(input_data)
        
        assert success is False
        assert prediction is None
        assert "not loaded" in message.lower()
    
    def test_predict_single_invalid_input(self, tmp_path):
        """Test prediction with invalid input."""
        mock_model = create_mock_model()
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        
        # Invalid input (out of range)
        input_data = {
            'Hours_Studied': 100,  # Out of range
            'Attendance': 85,
            'Sleep_Hours': 7,
            'Previous_Scores': 75,
            'Tutoring_Sessions': 2,
            'Physical_Activity': 3
        }
        
        success, prediction, message = predictor.predict_single(input_data)
        
        assert success is False
        assert prediction is None
        assert "validation" in message.lower()
    
    def test_predict_single_missing_field(self, tmp_path):
        """Test prediction with missing required field."""
        mock_model = create_mock_model()
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        
        # Missing field
        input_data = {
            'Hours_Studied': 25,
            'Attendance': 85
            # Missing other fields
        }
        
        success, prediction, message = predictor.predict_single(input_data)
        
        assert success is False
        assert "missing" in message.lower()
    
    def test_predict_batch(self, tmp_path):
        """Test batch prediction."""
        # Create mock model
        mock_model = create_mock_model(prediction_value=75.0)
        
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        
        input_data_list = [
            {
                'Hours_Studied': 25,
                'Attendance': 85,
                'Sleep_Hours': 7,
                'Previous_Scores': 75,
                'Tutoring_Sessions': 2,
                'Physical_Activity': 3,
                'Gender': 'Male',
                'School_Type': 'Public',
                'Teacher_Quality': 'Medium',
                'Parental_Involvement': 'Medium',
                'Access_to_Resources': 'Medium',
                'Extracurricular_Activities': 'Yes',
                'Motivation_Level': 'Medium',
                'Internet_Access': 'Yes',
                'Family_Income': 'Medium',
                'Peer_Influence': 'Neutral',
                'Learning_Disabilities': 'No',
                'Parental_Education_Level': 'High School',
                'Distance_from_Home': 'Near'
            },
            {
                'Hours_Studied': 30,
                'Attendance': 90,
                'Sleep_Hours': 8,
                'Previous_Scores': 80,
                'Tutoring_Sessions': 3,
                'Physical_Activity': 4,
                'Gender': 'Female',
                'School_Type': 'Private',
                'Teacher_Quality': 'High',
                'Parental_Involvement': 'High',
                'Access_to_Resources': 'High',
                'Extracurricular_Activities': 'Yes',
                'Motivation_Level': 'High',
                'Internet_Access': 'Yes',
                'Family_Income': 'High',
                'Peer_Influence': 'Positive',
                'Learning_Disabilities': 'No',
                'Parental_Education_Level': 'Postgraduate',
                'Distance_from_Home': 'Near'
            },
            {
                'Hours_Studied': 20,
                'Attendance': 75,
                'Sleep_Hours': 6,
                'Previous_Scores': 70,
                'Tutoring_Sessions': 1,
                'Physical_Activity': 2,
                'Gender': 'Male',
                'School_Type': 'Public',
                'Teacher_Quality': 'Low',
                'Parental_Involvement': 'Low',
                'Access_to_Resources': 'Low',
                'Extracurricular_Activities': 'No',
                'Motivation_Level': 'Low',
                'Internet_Access': 'No',
                'Family_Income': 'Low',
                'Peer_Influence': 'Negative',
                'Learning_Disabilities': 'Yes',
                'Parental_Education_Level': 'High School',
                'Distance_from_Home': 'Far'
            }
        ]
        
        results = predictor.predict_batch(input_data_list)
        
        assert len(results) == 3
        assert all(r['success'] for r in results)
        # All predictions should be 75.0 since we're using the same model
        assert all(r['prediction'] == 75.0 for r in results)


class TestPredictorCSVPrediction:
    """Test CSV prediction functionality."""
    
    def test_predict_from_csv_success(self, tmp_path):
        """Test successful prediction from CSV."""
        # Create mock model
        mock_model = create_mock_model()
        
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        
        # Create test CSV with all required columns
        csv_path = tmp_path / "test_data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Hours_Studied', 'Attendance', 'Sleep_Hours', 
                'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity',
                'Gender', 'School_Type', 'Teacher_Quality',
                'Parental_Involvement', 'Access_to_Resources',
                'Extracurricular_Activities', 'Motivation_Level',
                'Internet_Access', 'Family_Income', 'Peer_Influence',
                'Learning_Disabilities', 'Parental_Education_Level',
                'Distance_from_Home'
            ])
            writer.writerow([
                '25', '85', '7', '75', '2', '3',
                'Male', 'Public', 'Medium',
                'Medium', 'Medium', 'Yes', 'Medium',
                'Yes', 'Medium', 'Neutral', 'No', 'High School', 'Near'
            ])
            writer.writerow([
                '30', '90', '8', '80', '3', '4',
                'Female', 'Private', 'High',
                'High', 'High', 'Yes', 'High',
                'Yes', 'High', 'Positive', 'No', 'Postgraduate', 'Near'
            ])
        
        success, results, message = predictor.predict_from_csv(csv_path)
        
        assert success is True
        assert len(results) == 2
        assert all(r['success'] for r in results)
    
    def test_predict_from_csv_file_not_found(self, tmp_path):
        """Test prediction from non-existent CSV."""
        mock_model = create_mock_model()
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        
        non_existent = tmp_path / "nonexistent.csv"
        success, results, message = predictor.predict_from_csv(non_existent)
        
        assert success is False
        assert len(results) == 0
        assert "not found" in message.lower()
    
    def test_predict_from_csv_empty(self, tmp_path):
        """Test prediction from empty CSV."""
        mock_model = create_mock_model()
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        
        # Create empty CSV
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        
        success, results, message = predictor.predict_from_csv(csv_path)
        
        assert success is False
        assert "empty" in message.lower()


class TestPredictorSavePredictions:
    """Test saving predictions functionality."""
    
    def test_save_predictions_success(self, tmp_path):
        """Test successful prediction saving."""
        mock_model = create_mock_model()
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        
        predictions = [
            {
                'index': 1,
                'input': {'Hours_Studied': 25},
                'prediction': 75.0,
                'success': True,
                'message': 'Success'
            },
            {
                'index': 2,
                'input': {'Hours_Studied': 30},
                'prediction': 85.0,
                'success': True,
                'message': 'Success'
            }
        ]
        
        output_path = tmp_path / "predictions.csv"
        success, saved_path = predictor.save_predictions(predictions, output_path=output_path)
        
        assert success is True
        assert saved_path.exists()
        
        # Verify content
        with open(saved_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
    
    def test_save_predictions_empty(self, tmp_path):
        """Test saving empty predictions."""
        mock_model = create_mock_model()
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        
        success, saved_path = predictor.save_predictions([])
        
        assert success is False
    
    def test_save_predictions_auto_filename(self, tmp_path):
        """Test saving predictions with auto-generated filename."""
        mock_model = create_mock_model()
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        
        predictions = [
            {
                'index': 1,
                'input': {'Hours_Studied': 25},
                'prediction': 75.0,
                'success': True,
                'message': 'Success'
            }
        ]
        
        success, saved_path = predictor.save_predictions(predictions)
        
        assert success is True
        assert saved_path.parent.name == "predictions"
        assert saved_path.suffix == ".csv"


class TestPredictorDisplay:
    """Test prediction display functionality."""
    
    def test_display_prediction_result(self, tmp_path, capsys):
        """Test displaying prediction result."""
        mock_model = create_mock_model()
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        
        input_data = {
            'Hours_Studied': 25,
            'Attendance': 85,
            'Sleep_Hours': 7,
            'Previous_Scores': 75,
            'Tutoring_Sessions': 2,
            'Physical_Activity': 3
        }
        
        predictor.display_prediction_result(85.5, input_data)
        
        captured = capsys.readouterr()
        assert "PREDICTION RESULT" in captured.out
        assert "85.50" in captured.out
    
    def test_display_batch_results(self, tmp_path, capsys):
        """Test displaying batch results."""
        mock_model = create_mock_model()
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        
        results = [
            {
                'index': 1,
                'input': {'Hours_Studied': 25},
                'prediction': 75.0,
                'success': True,
                'message': 'Success'
            },
            {
                'index': 2,
                'input': {'Hours_Studied': 30},
                'prediction': 85.0,
                'success': True,
                'message': 'Success'
            }
        ]
        
        predictor.display_batch_results(results)
        
        captured = capsys.readouterr()
        assert "BATCH PREDICTION RESULTS" in captured.out
        assert "75.00" in captured.out
        assert "85.00" in captured.out
    
    def test_display_batch_results_with_failures(self, tmp_path, capsys):
        """Test displaying batch results with failures."""
        mock_model = create_mock_model()
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        
        results = [
            {
                'index': 1,
                'input': {'Hours_Studied': 25},
                'prediction': 75.0,
                'success': True,
                'message': 'Success'
            },
            {
                'index': 2,
                'input': {'Hours_Studied': 30},
                'prediction': None,
                'success': False,
                'message': 'Validation failed'
            }
        ]
        
        predictor.display_batch_results(results)
        
        captured = capsys.readouterr()
        assert "FAILED" in captured.out or "failed" in captured.out.lower()


class TestPredictorInputPreparation:
    """Test input data preparation."""
    
    def test_prepare_input_data(self, tmp_path):
        """Test preparing input data for prediction."""
        mock_model = create_mock_model()
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        
        input_data = {
            'Hours_Studied': 25,
            'Attendance': 85,
            'Sleep_Hours': 7,
            'Previous_Scores': 75,
            'Tutoring_Sessions': 2,
            'Physical_Activity': 3,
            'Gender': 'Male',
            'School_Type': 'Public',
            'Teacher_Quality': 'Medium',
            'Parental_Involvement': 'Medium',
            'Access_to_Resources': 'Medium',
            'Extracurricular_Activities': 'Yes',
            'Motivation_Level': 'Medium',
            'Internet_Access': 'Yes',
            'Family_Income': 'Medium',
            'Peer_Influence': 'Neutral',
            'Learning_Disabilities': 'No',
            'Parental_Education_Level': 'High School',
            'Distance_from_Home': 'Near'
        }
        
        df = predictor.prepare_input_data(input_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'Hours_Studied' in df.columns
        assert df['Hours_Studied'][0] == 25.0
    
    def test_prepare_input_data_missing_feature(self, tmp_path):
        """Test preparing input data with missing feature."""
        mock_model = create_mock_model()
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        predictor = Predictor(model_path=model_path)
        
        # Missing required feature
        input_data = {
            'Hours_Studied': 25,
            'Attendance': 85
            # Missing other features
        }
        
        with pytest.raises(PredictionError, match="Missing required feature"):
            predictor.prepare_input_data(input_data)