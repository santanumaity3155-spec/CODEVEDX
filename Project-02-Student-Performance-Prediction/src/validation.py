"""
Validation Module
Provides validation functions for user inputs and data integrity.
"""

import re
from typing import Tuple, Optional, List
from .config import VALIDATION_RANGES, CATEGORICAL_VALUES


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_numeric_input(
    value: str,
    field_name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_float: bool = True
) -> Tuple[bool, float, str]:
    """
    Validate numeric input from user.
    
    Args:
        value: String value to validate
        field_name: Name of the field (for error messages)
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)
        allow_float: Whether to allow float values
    
    Returns:
        Tuple of (is_valid, numeric_value, error_message)
    """
    # Check if empty
    if not value or not value.strip():
        return False, 0.0, f"{field_name} cannot be empty"
    
    # Try to convert to number
    try:
        if allow_float:
            numeric_value = float(value)
        else:
            numeric_value = int(value)
            if str(numeric_value) != value.strip():
                return False, 0.0, f"{field_name} must be a whole number"
    except ValueError:
        return False, 0.0, f"{field_name} must be a numeric value"
    
    # Check range
    if min_val is not None and numeric_value < min_val:
        return False, 0.0, f"{field_name} must be at least {min_val}"
    
    if max_val is not None and numeric_value > max_val:
        return False, 0.0, f"{field_name} must be at most {max_val}"
    
    return True, numeric_value, ""


def validate_hours_studied(value: str) -> Tuple[bool, float, str]:
    """
    Validate Hours Studied input.
    
    Args:
        value: String value to validate
    
    Returns:
        Tuple of (is_valid, numeric_value, error_message)
    """
    return validate_numeric_input(
        value,
        "Hours Studied",
        min_val=VALIDATION_RANGES['Hours_Studied']['min'],
        max_val=VALIDATION_RANGES['Hours_Studied']['max']
    )


def validate_attendance(value: str) -> Tuple[bool, float, str]:
    """
    Validate Attendance input.
    
    Args:
        value: String value to validate
    
    Returns:
        Tuple of (is_valid, numeric_value, error_message)
    """
    return validate_numeric_input(
        value,
        "Attendance",
        min_val=VALIDATION_RANGES['Attendance']['min'],
        max_val=VALIDATION_RANGES['Attendance']['max']
    )


def validate_sleep_hours(value: str) -> Tuple[bool, float, str]:
    """
    Validate Sleep Hours input.
    
    Args:
        value: String value to validate
    
    Returns:
        Tuple of (is_valid, numeric_value, error_message)
    """
    return validate_numeric_input(
        value,
        "Sleep Hours",
        min_val=VALIDATION_RANGES['Sleep_Hours']['min'],
        max_val=VALIDATION_RANGES['Sleep_Hours']['max']
    )


def validate_previous_scores(value: str) -> Tuple[bool, float, str]:
    """
    Validate Previous Scores input.
    
    Args:
        value: String value to validate
    
    Returns:
        Tuple of (is_valid, numeric_value, error_message)
    """
    return validate_numeric_input(
        value,
        "Previous Scores",
        min_val=VALIDATION_RANGES['Previous_Scores']['min'],
        max_val=VALIDATION_RANGES['Previous_Scores']['max']
    )


def validate_tutoring_sessions(value: str) -> Tuple[bool, float, str]:
    """
    Validate Tutoring Sessions input.
    
    Args:
        value: String value to validate
    
    Returns:
        Tuple of (is_valid, numeric_value, error_message)
    """
    return validate_numeric_input(
        value,
        "Tutoring Sessions",
        min_val=VALIDATION_RANGES['Tutoring_Sessions']['min'],
        max_val=VALIDATION_RANGES['Tutoring_Sessions']['max'],
        allow_float=False
    )


def validate_physical_activity(value: str) -> Tuple[bool, float, str]:
    """
    Validate Physical Activity input.
    
    Args:
        value: String value to validate
    
    Returns:
        Tuple of (is_valid, numeric_value, error_message)
    """
    return validate_numeric_input(
        value,
        "Physical Activity",
        min_val=VALIDATION_RANGES['Physical_Activity']['min'],
        max_val=VALIDATION_RANGES['Physical_Activity']['max'],
        allow_float=False
    )


def validate_gender(value: str) -> Tuple[bool, str, str]:
    """
    Validate Gender input.
    
    Args:
        value: String value to validate
    
    Returns:
        Tuple of (is_valid, validated_value, error_message)
    """
    if not value or not value.strip():
        return False, "", "Gender cannot be empty"
    
    value = value.strip().capitalize()
    
    if value in CATEGORICAL_VALUES['Gender']:
        return True, value, ""
    
    return False, "", f"Gender must be one of: {', '.join(CATEGORICAL_VALUES['Gender'])}"


def validate_school_type(value: str) -> Tuple[bool, str, str]:
    """
    Validate School Type input.
    
    Args:
        value: String value to validate
    
    Returns:
        Tuple of (is_valid, validated_value, error_message)
    """
    if not value or not value.strip():
        return False, "", "School Type cannot be empty"
    
    value = value.strip().capitalize()
    
    if value in CATEGORICAL_VALUES['School_Type']:
        return True, value, ""
    
    return False, "", f"School Type must be one of: {', '.join(CATEGORICAL_VALUES['School_Type'])}"


def validate_teacher_quality(value: str) -> Tuple[bool, str, str]:
    """
    Validate Teacher Quality input.
    
    Args:
        value: String value to validate
    
    Returns:
        Tuple of (is_valid, validated_value, error_message)
    """
    if not value or not value.strip():
        return False, "", "Teacher Quality cannot be empty"
    
    value = value.strip().capitalize()
    
    if value in CATEGORICAL_VALUES['Teacher_Quality']:
        return True, value, ""
    
    return False, "", f"Teacher Quality must be one of: {', '.join(CATEGORICAL_VALUES['Teacher_Quality'])}"


def validate_menu_choice(value: str, min_choice: int = 1, max_choice: int = 10) -> Tuple[bool, int, str]:
    """
    Validate menu choice input.
    
    Args:
        value: String value to validate
        min_choice: Minimum valid choice
        max_choice: Maximum valid choice
    
    Returns:
        Tuple of (is_valid, numeric_choice, error_message)
    """
    return validate_numeric_input(
        value,
        "Menu Choice",
        min_val=min_choice,
        max_val=max_choice,
        allow_float=False
    )


def validate_string_input(
    value: str,
    field_name: str,
    min_length: int = 1,
    max_length: int = 100,
    allow_empty: bool = False
) -> Tuple[bool, str, str]:
    """
    Validate string input.
    
    Args:
        value: String value to validate
        field_name: Name of the field (for error messages)
        min_length: Minimum length
        max_length: Maximum length
        allow_empty: Whether to allow empty strings
    
    Returns:
        Tuple of (is_valid, validated_value, error_message)
    """
    if not allow_empty and (not value or not value.strip()):
        return False, "", f"{field_name} cannot be empty"
    
    if allow_empty and not value:
        return True, "", ""
    
    value = value.strip()
    
    if len(value) < min_length:
        return False, "", f"{field_name} must be at least {min_length} characters"
    
    if len(value) > max_length:
        return False, "", f"{field_name} must be at most {max_length} characters"
    
    return True, value, ""


def validate_email(value: str) -> Tuple[bool, str, str]:
    """
    Validate email format.
    
    Args:
        value: Email string to validate
    
    Returns:
        Tuple of (is_valid, validated_value, error_message)
    """
    if not value or not value.strip():
        return False, "", "Email cannot be empty"
    
    value = value.strip().lower()
    
    # Basic email regex pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if re.match(email_pattern, value):
        return True, value, ""
    
    return False, "", "Invalid email format"


def validate_positive_integer(value: str, field_name: str) -> Tuple[bool, int, str]:
    """
    Validate positive integer input.
    
    Args:
        value: String value to validate
        field_name: Name of the field (for error messages)
    
    Returns:
        Tuple of (is_valid, numeric_value, error_message)
    """
    return validate_numeric_input(
        value,
        field_name,
        min_val=0,
        max_val=None,
        allow_float=False
    )


def validate_percentage(value: str, field_name: str) -> Tuple[bool, float, str]:
    """
    Validate percentage input (0-100).
    
    Args:
        value: String value to validate
        field_name: Name of the field (for error messages)
    
    Returns:
        Tuple of (is_valid, numeric_value, error_message)
    """
    return validate_numeric_input(
        value,
        field_name,
        min_val=0,
        max_val=100
    )


def validate_student_record(record: dict) -> Tuple[bool, List[str]]:
    """
    Validate a complete student record.
    
    Args:
        record: Dictionary containing student data
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Validate numeric fields
    numeric_fields = [
        ('Hours_Studied', validate_hours_studied),
        ('Attendance', validate_attendance),
        ('Sleep_Hours', validate_sleep_hours),
        ('Previous_Scores', validate_previous_scores),
        ('Tutoring_Sessions', validate_tutoring_sessions),
        ('Physical_Activity', validate_physical_activity)
    ]
    
    for field_name, validator in numeric_fields:
        if field_name in record:
            is_valid, _, error = validator(str(record[field_name]))
            if not is_valid:
                errors.append(error)
    
    # Validate categorical fields
    categorical_fields = [
        ('Gender', validate_gender),
        ('School_Type', validate_school_type),
        ('Teacher_Quality', validate_teacher_quality)
    ]
    
    for field_name, validator in categorical_fields:
        if field_name in record:
            is_valid, _, error = validator(str(record[field_name]))
            if not is_valid:
                errors.append(error)
    
    return len(errors) == 0, errors


def get_validated_input(
    prompt: str,
    validator,
    max_attempts: int = 3
) -> Tuple[bool, any, str]:
    """
    Get validated input from user with retry logic.
    
    Args:
        prompt: Input prompt to display
        validator: Validation function to use
        max_attempts: Maximum number of retry attempts
    
    Returns:
        Tuple of (success, validated_value, error_message)
    """
    attempts = 0
    
    while attempts < max_attempts:
        user_input = input(prompt).strip()
        
        is_valid, value, error = validator(user_input)
        
        if is_valid:
            return True, value, ""
        
        print(f"Invalid input: {error}")
        attempts += 1
    
    return False, None, f"Maximum attempts ({max_attempts}) exceeded"


def validate_search_query(value: str) -> Tuple[bool, str, str]:
    """
    Validate search query input.
    
    Args:
        value: Search query string
    
    Returns:
        Tuple of (is_valid, validated_value, error_message)
    """
    if not value or not value.strip():
        return False, "", "Search query cannot be empty"
    
    value = value.strip()
    
    if len(value) < 2:
        return False, "", "Search query must be at least 2 characters"
    
    if len(value) > 100:
        return False, "", "Search query must be at most 100 characters"
    
    return True, value, ""


def validate_export_format(value: str) -> Tuple[bool, str, str]:
    """
    Validate export format input.
    
    Args:
        value: Format string (csv, json, etc.)
    
    Returns:
        Tuple of (is_valid, validated_value, error_message)
    """
    if not value or not value.strip():
        return False, "", "Export format cannot be empty"
    
    value = value.strip().lower()
    
    valid_formats = ['csv', 'json', 'xlsx']
    
    if value in valid_formats:
        return True, value, ""
    
    return False, "", f"Export format must be one of: {', '.join(valid_formats)}"


def validate_file_path(value: str, must_exist: bool = False) -> Tuple[bool, str, str]:
    """
    Validate file path input.
    
    Args:
        value: File path string
        must_exist: Whether the file must already exist
    
    Returns:
        Tuple of (is_valid, validated_value, error_message)
    """
    from pathlib import Path
    
    if not value or not value.strip():
        return False, "", "File path cannot be empty"
    
    value = value.strip()
    path = Path(value)
    
    if must_exist and not path.exists():
        return False, "", f"File does not exist: {value}"
    
    # Check for invalid characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    if any(char in value for char in invalid_chars):
        return False, "", "File path contains invalid characters"
    
    return True, value, ""


def sanitize_input(value: str) -> str:
    """
    Sanitize user input by removing potentially dangerous characters.
    
    Args:
        value: Input string to sanitize
    
    Returns:
        Sanitized string
    """
    # Remove control characters except newline and tab
    sanitized = ''.join(char for char in value if char.isprintable() or char in ['\n', '\t'])
    
    # Strip leading/trailing whitespace
    sanitized = sanitized.strip()
    
    return sanitized


def validate_prediction_input(data: dict) -> Tuple[bool, List[str]]:
    """
    Validate input data for prediction.
    
    Args:
        data: Dictionary containing prediction input features
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    required_fields = [
        'Hours_Studied',
        'Attendance',
        'Sleep_Hours',
        'Previous_Scores',
        'Tutoring_Sessions',
        'Physical_Activity'
    ]
    
    # Check for missing fields
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Validate numeric fields
    if 'Hours_Studied' in data:
        is_valid, _, error = validate_hours_studied(str(data['Hours_Studied']))
        if not is_valid:
            errors.append(error)
    
    if 'Attendance' in data:
        is_valid, _, error = validate_attendance(str(data['Attendance']))
        if not is_valid:
            errors.append(error)
    
    if 'Sleep_Hours' in data:
        is_valid, _, error = validate_sleep_hours(str(data['Sleep_Hours']))
        if not is_valid:
            errors.append(error)
    
    if 'Previous_Scores' in data:
        is_valid, _, error = validate_previous_scores(str(data['Previous_Scores']))
        if not is_valid:
            errors.append(error)
    
    if 'Tutoring_Sessions' in data:
        is_valid, _, error = validate_tutoring_sessions(str(data['Tutoring_Sessions']))
        if not is_valid:
            errors.append(error)
    
    if 'Physical_Activity' in data:
        is_valid, _, error = validate_physical_activity(str(data['Physical_Activity']))
        if not is_valid:
            errors.append(error)
    
    return len(errors) == 0, errors