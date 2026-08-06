"""
Validation functions for the Fake News Detection Tool.
Validates user inputs, files, and data integrity.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional

from config import MIN_TEXT_LENGTH, MAX_TEXT_LENGTH, PROCESSED_DATASET_PATH


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_text_input(text: str) -> Tuple[bool, str]:
    """
    Validate news text input.
    
    Args:
        text: Input text to validate
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    # Check for None or empty
    if text is None:
        return False, "Input text is None"
    
    # Check for empty or whitespace only
    if not text or not text.strip():
        return False, "Input text is empty or contains only whitespace"
    
    # Check minimum length
    text_length = len(text.strip())
    if text_length < MIN_TEXT_LENGTH:
        return False, f"Text too short. Minimum length is {MIN_TEXT_LENGTH} characters (got {text_length})"
    
    # Check maximum length
    if text_length > MAX_TEXT_LENGTH:
        return False, f"Text too long. Maximum length is {MAX_TEXT_LENGTH:,} characters (got {text_length:,})"
    
    return True, "Valid text input"


def validate_file_path(file_path: str) -> Tuple[bool, str]:
    """
    Validate if a file path exists and is accessible.
    
    Args:
        file_path: Path to validate
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not file_path:
        return False, "File path is empty"
    
    path = Path(file_path)
    
    if not path.exists():
        return False, f"File not found: {file_path}"
    
    if not path.is_file():
        return False, f"Path is not a file: {file_path}"
    
    # Check file extension
    valid_extensions = ['.txt', '.csv', '.pkl']
    if path.suffix.lower() not in valid_extensions:
        return False, f"Invalid file type. Supported types: {', '.join(valid_extensions)}"
    
    # Try to read the file
    try:
        with open(path, 'r', encoding='utf-8') as f:
            f.read(1)
        return True, "File is valid and accessible"
    except PermissionError:
        return False, f"Permission denied: {file_path}"
    except UnicodeDecodeError:
        return False, f"File encoding error: {file_path}"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def validate_csv_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate a CSV file for batch prediction.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    # First check if file exists
    exists, message = validate_file_path(file_path)
    if not exists:
        return False, message
    
    path = Path(file_path)
    
    # Check extension
    if path.suffix.lower() != '.csv':
        return False, f"Invalid file type. Expected .csv, got {path.suffix}"
    
    # Try to read CSV
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        
        if df.empty:
            return False, "CSV file is empty"
        
        if len(df) == 0:
            return False, "CSV file contains no rows"
        
        return True, f"Valid CSV with {len(df)} rows and {len(df.columns)} columns"
    
    except pd.errors.EmptyDataError:
        return False, "CSV file is empty or has no columns"
    except pd.errors.ParserError as e:
        return False, f"CSV parsing error: {str(e)}"
    except Exception as e:
        return False, f"Error reading CSV: {str(e)}"


def validate_text_column(file_path: str, column_name: str) -> Tuple[bool, str]:
    """
    Validate if a text column exists in a CSV file.
    
    Args:
        file_path: Path to CSV file
        column_name: Name of the text column
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        
        if column_name not in df.columns:
            available = ', '.join(df.columns.tolist())
            return False, f"Column '{column_name}' not found. Available columns: {available}"
        
        # Check if column has any non-null values
        if df[column_name].isna().all():
            return False, f"Column '{column_name}' contains only empty values"
        
        return True, f"Column '{column_name}' found with {df[column_name].notna().sum()} non-null values"
    
    except Exception as e:
        return False, f"Error validating column: {str(e)}"


def validate_model_file(file_path: Path) -> Tuple[bool, str]:
    """
    Validate if a model file is valid and loadable.
    
    Args:
        file_path: Path to model file
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not file_path.exists():
        return False, f"Model file not found: {file_path}"
    
    try:
        import pickle
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        
        # Check if it has a predict method
        if not hasattr(model, 'predict'):
            return False, "Loaded object does not have a predict method"
        
        return True, "Model file is valid"
    
    except Exception as e:
        return False, f"Error loading model: {str(e)}"


def validate_dataset() -> Tuple[bool, str]:
    """
    Validate if the processed dataset exists and is valid.
    
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not PROCESSED_DATASET_PATH.exists():
        return False, f"Dataset not found at {PROCESSED_DATASET_PATH}"
    
    try:
        import pandas as pd
        df = pd.read_csv(PROCESSED_DATASET_PATH)
        
        if df.empty:
            return False, "Dataset is empty"
        
        required_columns = ['text', 'label']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            return False, f"Dataset missing required columns: {', '.join(missing)}"
        
        return True, f"Dataset valid with {len(df)} rows"
    
    except Exception as e:
        return False, f"Error reading dataset: {str(e)}"


def validate_menu_choice(choice: str, min_val: int, max_val: int) -> Tuple[bool, int, str]:
    """
    Validate menu choice input.
    
    Args:
        choice: User input string
        min_val: Minimum valid value
        max_val: Maximum valid value
        
    Returns:
        Tuple of (is_valid: bool, parsed_value: int, message: str)
    """
    try:
        value = int(choice.strip())
        if min_val <= value <= max_val:
            return True, value, "Valid choice"
        else:
            return False, 0, f"Please enter a number between {min_val} and {max_val}"
    except ValueError:
        return False, 0, "Invalid input. Please enter a number"


def validate_export_format(format_type: str) -> Tuple[bool, str]:
    """
    Validate export format.
    
    Args:
        format_type: Export format string
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    valid_formats = ['csv', 'json', 'txt']
    format_lower = format_type.lower().strip()
    
    if format_lower not in valid_formats:
        return False, f"Invalid format. Supported formats: {', '.join(valid_formats)}"
    
    return True, format_lower


def validate_history_limit(limit: str) -> Tuple[bool, int, str]:
    """
    Validate history display limit.
    
    Args:
        limit: User input string
        
    Returns:
        Tuple of (is_valid: bool, parsed_value: int, message: str)
    """
    try:
        value = int(limit.strip())
        if value <= 0:
            return False, 0, "Limit must be a positive number"
        if value > 1000:
            return False, 0, "Maximum limit is 1000"
        return True, value, "Valid limit"
    except ValueError:
        return False, 0, "Invalid input. Please enter a number"