"""
Validation Module for Utility Usage Prediction Tool

This module provides comprehensive input validation functions for the application.
It validates user inputs including integers, floats, menu choices, and strings.

Author: CodeVedX AI/ML Internship
"""

import re
from typing import Optional, Tuple, Union
from datetime import datetime

from config import (
    MIN_VOLTAGE, MAX_VOLTAGE,
    MIN_INTENSITY, MAX_INTENSITY,
    MIN_ACTIVE_POWER, MAX_ACTIVE_POWER
)


# ========================================
# BASIC VALIDATION FUNCTIONS
# ========================================


def validate_integer(
    value: str,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    field_name: str = "Value"
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Validate if input is a valid integer within specified range.
    
    Args:
        value: Input string to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        field_name: Name of the field for error messages
    
    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    
    Example:
        >>> valid, val, err = validate_integer("25", min_value=0, max_value=100)
        >>> if valid:
        ...     print(f"Valid integer: {val}")
    """
    # Check if empty
    if not value or not value.strip():
        return False, None, f"{field_name} cannot be empty"
    
    # Try to parse as integer
    try:
        int_value = int(value.strip())
    except ValueError:
        return False, None, f"{field_name} must be a valid integer (got '{value}')"
    
    # Check range
    if min_value is not None and int_value < min_value:
        return False, None, f"{field_name} must be >= {min_value} (got {int_value})"
    
    if max_value is not None and int_value > max_value:
        return False, None, f"{field_name} must be <= {max_value} (got {int_value})"
    
    return True, int_value, None


def validate_float(
    value: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    field_name: str = "Value",
    allow_negative: bool = False
) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Validate if input is a valid float within specified range.
    
    Args:
        value: Input string to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        field_name: Name of the field for error messages
        allow_negative: Whether negative values are allowed
    
    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    
    Example:
        >>> valid, val, err = validate_float("3.14", min_value=0.0, max_value=10.0)
        >>> if valid:
        ...     print(f"Valid float: {val}")
    """
    # Check if empty
    if not value or not value.strip():
        return False, None, f"{field_name} cannot be empty"
    
    # Try to parse as float
    try:
        float_value = float(value.strip())
    except ValueError:
        return False, None, f"{field_name} must be a valid number (got '{value}')"
    
    # Check if negative values are allowed
    if not allow_negative and float_value < 0:
        return False, None, f"{field_name} cannot be negative (got {float_value})"
    
    # Check range
    if min_value is not None and float_value < min_value:
        return False, None, f"{field_name} must be >= {min_value} (got {float_value})"
    
    if max_value is not None and float_value > max_value:
        return False, None, f"{field_name} must be <= {max_value} (got {float_value})"
    
    return True, float_value, None


def validate_menu_choice(
    value: str,
    min_choice: int = 1,
    max_choice: int = 10,
    field_name: str = "Menu choice"
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Validate if input is a valid menu choice.
    
    Args:
        value: Input string to validate
        min_choice: Minimum valid menu option
        max_choice: Maximum valid menu option
        field_name: Name of the field for error messages
    
    Returns:
        Tuple of (is_valid, parsed_choice, error_message)
    
    Example:
        >>> valid, choice, err = validate_menu_choice("3", min_choice=1, max_choice=10)
        >>> if valid:
        ...     print(f"Valid menu choice: {choice}")
    """
    return validate_integer(
        value=value,
        min_value=min_choice,
        max_value=max_choice,
        field_name=field_name
    )


def validate_string(
    value: str,
    min_length: int = 1,
    max_length: int = 100,
    field_name: str = "Value",
    allow_empty: bool = False,
    pattern: Optional[str] = None
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate if input is a valid string.
    
    Args:
        value: Input string to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        field_name: Name of the field for error messages
        allow_empty: Whether empty strings are allowed
        pattern: Optional regex pattern to match
    
    Returns:
        Tuple of (is_valid, cleaned_value, error_message)
    
    Example:
        >>> valid, val, err = validate_string("John", min_length=2, max_length=50)
        >>> if valid:
        ...     print(f"Valid string: {val}")
    """
    # Clean the input
    cleaned_value = value.strip() if value else ""
    
    # Check if empty
    if not cleaned_value and not allow_empty:
        return False, None, f"{field_name} cannot be empty"
    
    # Check length
    if len(cleaned_value) < min_length:
        return False, None, f"{field_name} must be at least {min_length} characters (got {len(cleaned_value)})"
    
    if len(cleaned_value) > max_length:
        return False, None, f"{field_name} must be at most {max_length} characters (got {len(cleaned_value)})"
    
    # Check pattern
    if pattern and not re.match(pattern, cleaned_value):
        return False, None, f"{field_name} format is invalid"
    
    return True, cleaned_value, None


def validate_negative_values(
    value: float,
    field_name: str = "Value",
    allow_negative: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validate if a numeric value is negative when not allowed.
    
    Args:
        value: Numeric value to validate
        field_name: Name of the field for error messages
        allow_negative: Whether negative values are allowed
    
    Returns:
        Tuple of (is_valid, error_message)
    
    Example:
        >>> valid, err = validate_negative_values(-5.0, field_name="Voltage", allow_negative=False)
        >>> if not valid:
        ...     print(f"Error: {err}")
    """
    if not allow_negative and value < 0:
        return False, f"{field_name} cannot be negative (got {value})"
    
    return True, None


# ========================================
# DOMAIN-SPECIFIC VALIDATION FUNCTIONS
# ========================================


def validate_voltage(value: str) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Validate voltage value.
    
    Args:
        value: Input string to validate
    
    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    valid, float_value, error = validate_float(
        value=value,
        min_value=MIN_VOLTAGE,
        max_value=MAX_VOLTAGE,
        field_name="Voltage"
    )
    
    if not valid:
        return False, None, f"Invalid voltage: {error}"
    
    return True, float_value, None


def validate_intensity(value: str) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Validate current intensity value.
    
    Args:
        value: Input string to validate
    
    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    valid, float_value, error = validate_float(
        value=value,
        min_value=MIN_INTENSITY,
        max_value=MAX_INTENSITY,
        field_name="Intensity"
    )
    
    if not valid:
        return False, None, f"Invalid intensity: {error}"
    
    return True, float_value, None


def validate_active_power(value: str) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Validate active power value.
    
    Args:
        value: Input string to validate
    
    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    valid, float_value, error = validate_float(
        value=value,
        min_value=MIN_ACTIVE_POWER,
        max_value=MAX_ACTIVE_POWER,
        field_name="Active Power"
    )
    
    if not valid:
        return False, None, f"Invalid active power: {error}"
    
    return True, float_value, None


def validate_year(value: str) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Validate year value.
    
    Args:
        value: Input string to validate
    
    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    current_year = datetime.now().year
    
    valid, int_value, error = validate_integer(
        value=value,
        min_value=2000,
        max_value=current_year,
        field_name="Year"
    )
    
    if not valid:
        return False, None, f"Invalid year: {error}"
    
    return True, int_value, None


def validate_month(value: str) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Validate month value.
    
    Args:
        value: Input string to validate
    
    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    valid, int_value, error = validate_integer(
        value=value,
        min_value=1,
        max_value=12,
        field_name="Month"
    )
    
    if not valid:
        return False, None, f"Invalid month: {error}"
    
    return True, int_value, None


def validate_day(value: str, month: int, year: int) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Validate day value based on month and year.
    
    Args:
        value: Input string to validate
        month: Month number (1-12)
        year: Year number
    
    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    # Determine max days in month
    if month in [4, 6, 9, 11]:
        max_day = 30
    elif month == 2:
        # Check for leap year
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            max_day = 29
        else:
            max_day = 28
    else:
        max_day = 31
    
    valid, int_value, error = validate_integer(
        value=value,
        min_value=1,
        max_value=max_day,
        field_name="Day"
    )
    
    if not valid:
        return False, None, f"Invalid day: {error}"
    
    return True, int_value, None


def validate_hour(value: str) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Validate hour value.
    
    Args:
        value: Input string to validate
    
    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    valid, int_value, error = validate_integer(
        value=value,
        min_value=0,
        max_value=23,
        field_name="Hour"
    )
    
    if not valid:
        return False, None, f"Invalid hour: {error}"
    
    return True, int_value, None


def validate_record_id(value: str, existing_ids: set) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Validate record ID.
    
    Args:
        value: Input string to validate
        existing_ids: Set of existing record IDs
    
    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    valid, int_value, error = validate_integer(
        value=value,
        min_value=1,
        field_name="Record ID"
    )
    
    if not valid:
        return False, None, f"Invalid record ID: {error}"
    
    if int_value not in existing_ids:
        return False, None, f"Record ID {int_value} does not exist"
    
    return True, int_value, None


# ========================================
# INPUT HELPER FUNCTIONS
# ========================================


def get_valid_input(
    prompt: str,
    validation_func,
    max_attempts: int = 3,
    **validation_kwargs
) -> Tuple[bool, Optional[Union[int, float, str]], Optional[str]]:
    """
    Get valid input from user with retry logic.
    
    Args:
        prompt: Input prompt to display
        validation_func: Validation function to use
        max_attempts: Maximum number of retry attempts
        **validation_kwargs: Additional arguments for validation function
    
    Returns:
        Tuple of (success, validated_value, error_message)
    
    Example:
        >>> valid, value, err = get_valid_input(
        ...     "Enter voltage: ",
        ...     validate_voltage,
        ...     max_attempts=3
        ... )
    """
    attempts = 0
    
    while attempts < max_attempts:
        user_input = input(prompt).strip()
        
        is_valid, parsed_value, error = validation_func(user_input, **validation_kwargs)
        
        if is_valid:
            return True, parsed_value, None
        
        attempts += 1
        remaining = max_attempts - attempts
        
        if remaining > 0:
            print(f"  ✗ {error}. {remaining} attempt(s) remaining.")
        else:
            print(f"  ✗ {error}. No attempts remaining.")
            return False, None, error
    
    return False, None, "Maximum attempts exceeded"


def confirm_action(prompt: str = "Are you sure? (y/n): ") -> bool:
    """
    Get confirmation from user for an action.
    
    Args:
        prompt: Confirmation prompt to display
    
    Returns:
        True if user confirms, False otherwise
    """
    while True:
        response = input(prompt).strip().lower()
        
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("  ✗ Please enter 'y' or 'n'")


# ========================================
# VALIDATION UTILITIES
# ========================================


def is_valid_email(email: str) -> bool:
    """
    Check if string is a valid email address.
    
    Args:
        email: Email string to validate
    
    Returns:
        True if valid email, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_phone(phone: str) -> bool:
    """
    Check if string is a valid phone number.
    
    Args:
        phone: Phone string to validate
    
    Returns:
        True if valid phone, False otherwise
    """
    # Remove common separators and leading +
    cleaned = re.sub(r'[\s\-\(\)\+]', '', phone)
    # Check if it's 10-15 digits
    return bool(re.match(r'^\d{10,15}$', cleaned))


def sanitize_string(value: str, max_length: int = 100) -> str:
    """
    Sanitize string by removing special characters and limiting length.
    
    Args:
        value: String to sanitize
        max_length: Maximum length of sanitized string
    
    Returns:
        Sanitized string
    """
    # Remove special characters except spaces, hyphens, and underscores
    sanitized = re.sub(r'[^a-zA-Z0-9\s\-_]', '', value)
    # Limit length
    return sanitized[:max_length].strip()