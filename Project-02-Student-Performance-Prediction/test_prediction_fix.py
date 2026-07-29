#!/usr/bin/env python3
"""
Test script to verify the prediction fix works correctly.
This tests the complete prediction workflow without requiring interactive input.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH, DATASET_PATH
from src.data_handler import DataHandler
from src.predictor import Predictor

def test_prediction_workflow():
    """Test the complete prediction workflow."""
    print("=" * 80)
    print("TESTING PREDICTION WORKFLOW")
    print("=" * 80)
    
    # Test 1: Initialize DataHandler
    print("\n1. Testing DataHandler initialization...")
    try:
        data_handler = DataHandler()
        record_count = data_handler.get_record_count()
        print(f"   ✓ DataHandler loaded {record_count} records")
    except Exception as e:
        print(f"   ✗ DataHandler failed: {e}")
        return False
    
    # Test 2: Initialize Predictor
    print("\n2. Testing Predictor initialization...")
    try:
        predictor = Predictor()
        is_ready = predictor.is_model_ready()
        print(f"   ✓ Predictor initialized, model ready: {is_ready}")
        
        if not is_ready:
            print(f"   ✗ Model not ready. Expected path: {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"   ✗ Predictor failed: {e}")
        return False
    
    # Test 3: Test prediction with sample data
    print("\n3. Testing single prediction...")
    
    # Sample input with all required fields
    sample_input = {
        'Hours_Studied': 6.0,
        'Attendance': 76.0,
        'Sleep_Hours': 6.0,
        'Previous_Scores': 67.0,
        'Tutoring_Sessions': 6,
        'Physical_Activity': 6,
        'Gender': 'Male',
        'School_Type': 'Private',
        'Teacher_Quality': 'Medium',
        'Parental_Involvement': 'Medium',
        'Access_to_Resources': 'Medium',
        'Extracurricular_Activities': 'No',
        'Motivation_Level': 'Medium',
        'Internet_Access': 'Yes',
        'Family_Income': 'Medium',
        'Peer_Influence': 'Positive',
        'Learning_Disabilities': 'No',
        'Parental_Education_Level': 'College',
        'Distance_from_Home': 'Near'
    }
    
    try:
        success, prediction, message = predictor.predict_single(sample_input)
        
        if success and prediction is not None:
            print(f"   ✓ Prediction successful!")
            print(f"   ✓ Predicted score: {prediction:.2f}")
            print(f"   ✓ Message: {message}")
        else:
            print(f"   ✗ Prediction failed: {message}")
            return False
    except Exception as e:
        print(f"   ✗ Prediction raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test with another sample
    print("\n4. Testing with different input values...")
    sample_input2 = {
        'Hours_Studied': 23.0,
        'Attendance': 84.0,
        'Sleep_Hours': 7.0,
        'Previous_Scores': 73.0,
        'Tutoring_Sessions': 0,
        'Physical_Activity': 3,
        'Gender': 'Male',
        'School_Type': 'Public',
        'Teacher_Quality': 'Medium',
        'Parental_Involvement': 'Low',
        'Access_to_Resources': 'High',
        'Extracurricular_Activities': 'No',
        'Motivation_Level': 'Low',
        'Internet_Access': 'Yes',
        'Family_Income': 'Low',
        'Peer_Influence': 'Positive',
        'Learning_Disabilities': 'No',
        'Parental_Education_Level': 'High School',
        'Distance_from_Home': 'Near'
    }
    
    try:
        success, prediction, message = predictor.predict_single(sample_input2)
        
        if success and prediction is not None:
            print(f"   ✓ Prediction successful!")
            print(f"   ✓ Predicted score: {prediction:.2f}")
        else:
            print(f"   ✗ Prediction failed: {message}")
            return False
    except Exception as e:
        print(f"   ✗ Prediction raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Verify all validation functions exist
    print("\n5. Testing validation functions...")
    try:
        from src.validation import (
            validate_hours_studied, validate_attendance, validate_sleep_hours,
            validate_previous_scores, validate_tutoring_sessions, validate_physical_activity,
            validate_gender, validate_school_type, validate_teacher_quality,
            validate_parental_involvement, validate_access_to_resources,
            validate_extracurricular_activities, validate_motivation_level,
            validate_internet_access, validate_family_income, validate_peer_influence,
            validate_learning_disabilities, validate_parental_education_level,
            validate_distance_from_home
        )
        print("   ✓ All validation functions imported successfully")
        
        # Test a few validators
        success, val, error = validate_gender("Male")
        if success and val == "Male":
            print("   ✓ validate_gender works")
        else:
            print(f"   ✗ validate_gender failed: {error}")
            return False
        
        success, val, error = validate_school_type("Public")
        if success and val == "Public":
            print("   ✓ validate_school_type works")
        else:
            print(f"   ✗ validate_school_type failed: {error}")
            return False
            
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
    return True

if __name__ == "__main__":
    try:
        success = test_prediction_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)