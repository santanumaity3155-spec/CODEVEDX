"""
Tests for Validation Module
"""

import pytest
from src.validation import (
    validate_numeric_input, validate_hours_studied, validate_attendance,
    validate_sleep_hours, validate_previous_scores, validate_tutoring_sessions,
    validate_physical_activity, validate_gender, validate_school_type,
    validate_teacher_quality, validate_menu_choice, validate_string_input,
    validate_email, validate_positive_integer, validate_percentage,
    validate_student_record, validate_search_query, validate_export_format,
    validate_file_path, sanitize_input, validate_prediction_input,
    ValidationError
)


class TestNumericValidation:
    """Test numeric input validation."""
    
    def test_validate_numeric_input_valid_integer(self):
        """Test valid integer input."""
        is_valid, value, error = validate_numeric_input("25", "Test", allow_float=False)
        assert is_valid
        assert value == 25
        assert error == ""
    
    def test_validate_numeric_input_valid_float(self):
        """Test valid float input."""
        is_valid, value, error = validate_numeric_input("25.5", "Test")
        assert is_valid
        assert value == 25.5
        assert error == ""
    
    def test_validate_numeric_input_empty(self):
        """Test empty input."""
        is_valid, value, error = validate_numeric_input("", "Test")
        assert not is_valid
        assert "cannot be empty" in error
    
    def test_validate_numeric_input_non_numeric(self):
        """Test non-numeric input."""
        is_valid, value, error = validate_numeric_input("abc", "Test")
        assert not is_valid
        assert "must be a numeric value" in error
    
    def test_validate_numeric_input_below_min(self):
        """Test value below minimum."""
        is_valid, value, error = validate_numeric_input("5", "Test", min_val=10, max_val=100)
        assert not is_valid
        assert "at least 10" in error
    
    def test_validate_numeric_input_above_max(self):
        """Test value above maximum."""
        is_valid, value, error = validate_numeric_input("150", "Test", min_val=0, max_val=100)
        assert not is_valid
        assert "at most 100" in error
    
    def test_validate_numeric_input_within_range(self):
        """Test value within range."""
        is_valid, value, error = validate_numeric_input("50", "Test", min_val=0, max_val=100)
        assert is_valid
        assert value == 50
        assert error == ""


class TestFieldSpecificValidation:
    """Test field-specific validation functions."""
    
    def test_validate_hours_studied_valid(self):
        """Test valid hours studied."""
        is_valid, value, error = validate_hours_studied("25")
        assert is_valid
        assert value == 25.0
    
    def test_validate_hours_studied_out_of_range(self):
        """Test hours studied out of range."""
        is_valid, value, error = validate_hours_studied("100")
        assert not is_valid
        assert "at most 50" in error
    
    def test_validate_attendance_valid(self):
        """Test valid attendance."""
        is_valid, value, error = validate_attendance("85")
        assert is_valid
        assert value == 85.0
    
    def test_validate_attendance_out_of_range(self):
        """Test attendance out of range."""
        is_valid, value, error = validate_attendance("150")
        assert not is_valid
        assert "at most 100" in error
    
    def test_validate_sleep_hours_valid(self):
        """Test valid sleep hours."""
        is_valid, value, error = validate_sleep_hours("7")
        assert is_valid
        assert value == 7.0
    
    def test_validate_sleep_hours_out_of_range(self):
        """Test sleep hours out of range."""
        is_valid, value, error = validate_sleep_hours("30")
        assert not is_valid
        assert "at most 24" in error
    
    def test_validate_previous_scores_valid(self):
        """Test valid previous scores."""
        is_valid, value, error = validate_previous_scores("75")
        assert is_valid
        assert value == 75.0
    
    def test_validate_tutoring_sessions_valid(self):
        """Test valid tutoring sessions."""
        is_valid, value, error = validate_tutoring_sessions("3")
        assert is_valid
        assert value == 3
    
    def test_validate_tutoring_sessions_float_rejected(self):
        """Test that tutoring sessions rejects floats."""
        is_valid, value, error = validate_tutoring_sessions("3.5")
        assert not is_valid
        assert "whole number" in error
    
    def test_validate_physical_activity_valid(self):
        """Test valid physical activity."""
        is_valid, value, error = validate_physical_activity("5")
        assert is_valid
        assert value == 5


class TestCategoricalValidation:
    """Test categorical input validation."""
    
    def test_validate_gender_male(self):
        """Test valid male gender."""
        is_valid, value, error = validate_gender("male")
        assert is_valid
        assert value == "Male"
        assert error == ""
    
    def test_validate_gender_female(self):
        """Test valid female gender."""
        is_valid, value, error = validate_gender("FEMALE")
        assert is_valid
        assert value == "Female"
        assert error == ""
    
    def test_validate_gender_invalid(self):
        """Test invalid gender."""
        is_valid, value, error = validate_gender("other")
        assert not is_valid
        assert "must be one of" in error
    
    def test_validate_gender_empty(self):
        """Test empty gender."""
        is_valid, value, error = validate_gender("")
        assert not is_valid
        assert "cannot be empty" in error
    
    def test_validate_school_type_public(self):
        """Test valid public school type."""
        is_valid, value, error = validate_school_type("public")
        assert is_valid
        assert value == "Public"
    
    def test_validate_school_type_private(self):
        """Test valid private school type."""
        is_valid, value, error = validate_school_type("PRIVATE")
        assert is_valid
        assert value == "Private"
    
    def test_validate_school_type_invalid(self):
        """Test invalid school type."""
        is_valid, value, error = validate_school_type("charter")
        assert not is_valid
        assert "must be one of" in error
    
    def test_validate_teacher_quality_valid(self):
        """Test valid teacher quality."""
        for quality in ["low", "MEDIUM", "High"]:
            is_valid, value, error = validate_teacher_quality(quality)
            assert is_valid
            assert value in ["Low", "Medium", "High"]
    
    def test_validate_teacher_quality_invalid(self):
        """Test invalid teacher quality."""
        is_valid, value, error = validate_teacher_quality("excellent")
        assert not is_valid
        assert "must be one of" in error


class TestMenuChoiceValidation:
    """Test menu choice validation."""
    
    def test_validate_menu_choice_valid(self):
        """Test valid menu choice."""
        is_valid, value, error = validate_menu_choice("5", 1, 10)
        assert is_valid
        assert value == 5
    
    def test_validate_menu_choice_min_boundary(self):
        """Test menu choice at minimum boundary."""
        is_valid, value, error = validate_menu_choice("1", 1, 10)
        assert is_valid
        assert value == 1
    
    def test_validate_menu_choice_max_boundary(self):
        """Test menu choice at maximum boundary."""
        is_valid, value, error = validate_menu_choice("10", 1, 10)
        assert is_valid
        assert value == 10
    
    def test_validate_menu_choice_below_min(self):
        """Test menu choice below minimum."""
        is_valid, value, error = validate_menu_choice("0", 1, 10)
        assert not is_valid
        assert "at least 1" in error
    
    def test_validate_menu_choice_above_max(self):
        """Test menu choice above maximum."""
        is_valid, value, error = validate_menu_choice("11", 1, 10)
        assert not is_valid
        assert "at most 10" in error
    
    def test_validate_menu_choice_float_rejected(self):
        """Test that menu choice rejects floats."""
        is_valid, value, error = validate_menu_choice("5.5", 1, 10)
        assert not is_valid
        assert "whole number" in error


class TestStringValidation:
    """Test string input validation."""
    
    def test_validate_string_input_valid(self):
        """Test valid string input."""
        is_valid, value, error = validate_string_input("hello", "Test")
        assert is_valid
        assert value == "hello"
    
    def test_validate_string_input_empty_not_allowed(self):
        """Test empty string not allowed."""
        is_valid, value, error = validate_string_input("", "Test")
        assert not is_valid
        assert "cannot be empty" in error
    
    def test_validate_string_input_empty_allowed(self):
        """Test empty string allowed."""
        is_valid, value, error = validate_string_input("", "Test", allow_empty=True)
        assert is_valid
        assert value == ""
    
    def test_validate_string_input_too_short(self):
        """Test string too short."""
        is_valid, value, error = validate_string_input("ab", "Test", min_length=3)
        assert not is_valid
        assert "at least 3 characters" in error
    
    def test_validate_string_input_too_long(self):
        """Test string too long."""
        is_valid, value, error = validate_string_input("a" * 200, "Test", max_length=100)
        assert not is_valid
        assert "at most 100 characters" in error
    
    def test_validate_string_input_whitespace_stripped(self):
        """Test that whitespace is stripped."""
        is_valid, value, error = validate_string_input("  hello  ", "Test")
        assert is_valid
        assert value == "hello"


class TestEmailValidation:
    """Test email validation."""
    
    def test_validate_email_valid(self):
        """Test valid email."""
        is_valid, value, error = validate_email("test@example.com")
        assert is_valid
        assert value == "test@example.com"
    
    def test_validate_email_invalid(self):
        """Test invalid email."""
        is_valid, value, error = validate_email("invalid-email")
        assert not is_valid
        assert "Invalid email format" in error
    
    def test_validate_email_empty(self):
        """Test empty email."""
        is_valid, value, error = validate_email("")
        assert not is_valid
        assert "cannot be empty" in error


class TestSpecializedValidation:
    """Test specialized validation functions."""
    
    def test_validate_positive_integer_valid(self):
        """Test valid positive integer."""
        is_valid, value, error = validate_positive_integer("5", "Test")
        assert is_valid
        assert value == 5
    
    def test_validate_positive_integer_negative(self):
        """Test negative integer."""
        is_valid, value, error = validate_positive_integer("-5", "Test")
        assert not is_valid
        assert "at least 0" in error
    
    def test_validate_percentage_valid(self):
        """Test valid percentage."""
        is_valid, value, error = validate_percentage("75", "Test")
        assert is_valid
        assert value == 75.0
    
    def test_validate_percentage_out_of_range(self):
        """Test percentage out of range."""
        is_valid, value, error = validate_percentage("150", "Test")
        assert not is_valid
        assert "at most 100" in error
    
    def test_validate_search_query_valid(self):
        """Test valid search query."""
        is_valid, value, error = validate_search_query("john")
        assert is_valid
        assert value == "john"
    
    def test_validate_search_query_too_short(self):
        """Test search query too short."""
        is_valid, value, error = validate_search_query("a")
        assert not is_valid
        assert "at least 2 characters" in error
    
    def test_validate_export_format_valid(self):
        """Test valid export format."""
        for fmt in ["csv", "json", "xlsx"]:
            is_valid, value, error = validate_export_format(fmt)
            assert is_valid
            assert value == fmt
    
    def test_validate_export_format_invalid(self):
        """Test invalid export format."""
        is_valid, value, error = validate_export_format("xml")
        assert not is_valid
        assert "must be one of" in error
    
    def test_validate_file_path_valid(self):
        """Test valid file path."""
        is_valid, value, error = validate_file_path("data/file.csv")
        assert is_valid
        assert value == "data/file.csv"
    
    def test_validate_file_path_invalid_chars(self):
        """Test file path with invalid characters."""
        is_valid, value, error = validate_file_path("file<name>.csv")
        assert not is_valid
        assert "invalid characters" in error


class TestSanitizeInput:
    """Test input sanitization."""
    
    def test_sanitize_input_normal(self):
        """Test normal input."""
        result = sanitize_input("hello world")
        assert result == "hello world"
    
    def test_sanitize_input_whitespace(self):
        """Test input with whitespace."""
        result = sanitize_input("  hello  ")
        assert result == "hello"
    
    def test_sanitize_input_control_chars(self):
        """Test input with control characters."""
        result = sanitize_input("hello\x00\x01world")
        assert result == "helloworld"


class TestStudentRecordValidation:
    """Test student record validation."""
    
    def test_validate_student_record_valid(self):
        """Test valid student record."""
        record = {
            'Hours_Studied': 25,
            'Attendance': 85,
            'Sleep_Hours': 7,
            'Previous_Scores': 75,
            'Tutoring_Sessions': 2,
            'Physical_Activity': 3,
            'Gender': 'Male',
            'School_Type': 'Public',
            'Teacher_Quality': 'Medium'
        }
        
        is_valid, errors = validate_student_record(record)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_student_record_invalid_numeric(self):
        """Test student record with invalid numeric field."""
        record = {
            'Hours_Studied': 100,  # Out of range
            'Attendance': 85,
            'Sleep_Hours': 7,
            'Previous_Scores': 75,
            'Tutoring_Sessions': 2,
            'Physical_Activity': 3,
            'Gender': 'Male',
            'School_Type': 'Public',
            'Teacher_Quality': 'Medium'
        }
        
        is_valid, errors = validate_student_record(record)
        assert not is_valid
        assert len(errors) > 0
    
    def test_validate_student_record_invalid_categorical(self):
        """Test student record with invalid categorical field."""
        record = {
            'Hours_Studied': 25,
            'Attendance': 85,
            'Sleep_Hours': 7,
            'Previous_Scores': 75,
            'Tutoring_Sessions': 2,
            'Physical_Activity': 3,
            'Gender': 'Invalid',  # Invalid
            'School_Type': 'Public',
            'Teacher_Quality': 'Medium'
        }
        
        is_valid, errors = validate_student_record(record)
        assert not is_valid
        assert len(errors) > 0


class TestPredictionInputValidation:
    """Test prediction input validation."""
    
    def test_validate_prediction_input_valid(self):
        """Test valid prediction input."""
        data = {
            'Hours_Studied': 25,
            'Attendance': 85,
            'Sleep_Hours': 7,
            'Previous_Scores': 75,
            'Tutoring_Sessions': 2,
            'Physical_Activity': 3
        }
        
        is_valid, errors = validate_prediction_input(data)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_prediction_input_missing_field(self):
        """Test prediction input with missing field."""
        data = {
            'Hours_Studied': 25,
            'Attendance': 85,
            # Missing other fields
        }
        
        is_valid, errors = validate_prediction_input(data)
        assert not is_valid
        assert len(errors) > 0
        assert any("Missing required field" in error for error in errors)
    
    def test_validate_prediction_input_invalid_value(self):
        """Test prediction input with invalid value."""
        data = {
            'Hours_Studied': 100,  # Out of range
            'Attendance': 85,
            'Sleep_Hours': 7,
            'Previous_Scores': 75,
            'Tutoring_Sessions': 2,
            'Physical_Activity': 3
        }
        
        is_valid, errors = validate_prediction_input(data)
        assert not is_valid
        assert len(errors) > 0