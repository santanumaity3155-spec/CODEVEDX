"""
Utilities Module
Provides helper functions for formatting, directory creation, CSV export, 
date & time operations, and pretty console output.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import PREDICTIONS_DIR, OUTPUTS_DIR


def format_currency(amount: float) -> str:
    """
    Format a number as currency.
    
    Args:
        amount: Numeric value to format
    
    Returns:
        Formatted currency string
    """
    return f"${amount:,.2f}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format a number as percentage.
    
    Args:
        value: Numeric value (0-100)
        decimal_places: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimal_places}f}%"


def format_score(score: float, max_score: float = 100.0) -> str:
    """
    Format a score with grade indicator.
    
    Args:
        score: Score value
        max_score: Maximum possible score
    
    Returns:
        Formatted score string with grade
    """
    percentage = (score / max_score) * 100
    
    if percentage >= 90:
        grade = "A+"
    elif percentage >= 80:
        grade = "A"
    elif percentage >= 70:
        grade = "B"
    elif percentage >= 60:
        grade = "C"
    elif percentage >= 50:
        grade = "D"
    else:
        grade = "F"
    
    return f"{score:.2f}/{max_score} ({grade})"


def get_timestamp() -> str:
    """
    Get current timestamp in formatted string.
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_filename_timestamp() -> str:
    """
    Get timestamp suitable for filenames.
    
    Returns:
        Filename-safe timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_directory(directory_path: Path) -> bool:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory_path: Path to create
    
    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")
        return False


def ensure_directories() -> None:
    """
    Ensure all required directories exist.
    """
    directories = [
        PREDICTIONS_DIR,
        OUTPUTS_DIR
    ]
    
    for directory in directories:
        create_directory(directory)


def export_to_csv(
    data: List[Dict[str, Any]],
    file_path: Path,
    fieldnames: Optional[List[str]] = None
) -> bool:
    """
    Export data to CSV file.
    
    Args:
        data: List of dictionaries to export
        file_path: Path to save CSV file
        fieldnames: List of field names (uses keys from first dict if None)
    
    Returns:
        True if export successful, False otherwise
    """
    try:
        if not data:
            print("No data to export")
            return False
        
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get fieldnames from first record if not provided
        if fieldnames is None:
            fieldnames = list(data[0].keys())
        
        # Filter data to only include specified fieldnames
        filtered_data = []
        for record in data:
            filtered_record = {key: record.get(key, '') for key in fieldnames if key in record}
            filtered_data.append(filtered_record)
        
        # Write CSV
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_data)
        
        return True
    
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return False


def read_csv_file(file_path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    """
    Read CSV file and return data and headers.
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        Tuple of (list of records as dicts, list of headers)
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CSV is corrupted or empty
    """
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            if reader.fieldnames is None:
                raise ValueError("CSV file is empty or has no headers")
            
            headers = list(reader.fieldnames)
            records = list(reader)
            
            return records, headers
    
    except csv.Error as e:
        raise ValueError(f"Corrupted CSV file: {e}")


def print_separator(char: str = "=", length: int = 80) -> None:
    """
    Print a separator line.
    
    Args:
        char: Character to use for separator
        length: Length of separator
    """
    print(char * length)


def print_header(title: str, char: str = "=", length: int = 80) -> None:
    """
    Print a formatted header.
    
    Args:
        title: Header title
        char: Character to use for borders
        length: Total length of header
    """
    print_separator(char, length)
    # Center the title
    padding = (length - len(title) - 2) // 2
    remaining = length - len(title) - padding - 2
    line = f"{char}{' ' * padding} {title} {' ' * remaining}{char}"
    # Ensure the line is exactly the right length
    if len(line) < length:
        line = line + char * (length - len(line))
    elif len(line) > length:
        line = line[:length]
    print(line)
    print_separator(char, length)


def print_subheader(title: str, char: str = "-", length: int = 80) -> None:
    """
    Print a formatted subheader.
    
    Args:
        title: Subheader title
        char: Character to use for borders
        length: Total length of subheader
    """
    print_separator(char, length)
    print(f"  {title}")
    print_separator(char, length)


def print_success(message: str) -> None:
    """
    Print success message with checkmark.
    
    Args:
        message: Success message
    """
    print(f"✓ {message}")


def print_error(message: str) -> None:
    """
    Print error message with X mark.
    
    Args:
        message: Error message
    """
    print(f"✗ {message}")


def print_warning(message: str) -> None:
    """
    Print warning message with exclamation mark.
    
    Args:
        message: Warning message
    """
    print(f"⚠ {message}")


def print_info(message: str) -> None:
    """
    Print info message with info icon.
    
    Args:
        message: Info message
    """
    print(f"ℹ {message}")


def print_table(
    headers: List[str],
    rows: List[List[str]],
    padding: int = 2
) -> None:
    """
    Print data in a formatted table.
    
    Args:
        headers: List of column headers
        rows: List of rows (each row is a list of values)
        padding: Padding between columns
    """
    if not headers or not rows:
        print("No data to display")
        return
    
    # Calculate column widths
    col_widths = [len(str(header)) for header in headers]
    
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Print header
    header_row = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_row)
    print("-" * len(header_row))
    
    # Print rows
    for row in rows:
        row_str = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        print(row_str)


def get_user_confirmation(prompt: str = "Do you want to continue?") -> bool:
    """
    Get yes/no confirmation from user.
    
    Args:
        prompt: Confirmation prompt
    
    Returns:
        True if user confirms, False otherwise
    """
    while True:
        response = input(f"{prompt} (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")


def clear_screen() -> None:
    """
    Clear the console screen.
    """
    os.system('cls' if os.name == 'nt' else 'clear')


def pause_screen(message: str = "Press Enter to continue...") -> None:
    """
    Pause execution until user presses Enter.
    
    Args:
        message: Message to display
    """
    input(message)


def validate_range(
    value: float,
    min_val: float,
    max_val: float,
    field_name: str
) -> tuple[bool, str]:
    """
    Validate if a value is within a specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Name of the field (for error message)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if value < min_val or value > max_val:
        return False, f"{field_name} must be between {min_val} and {max_val}"
    return True, ""


def truncate_string(text: str, max_length: int = 50) -> str:
    """
    Truncate string to maximum length with ellipsis.
    
    Args:
        text: String to truncate
        max_length: Maximum length
    
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values.
    
    Args:
        values: List of numeric values
    
    Returns:
        Dictionary with statistics (mean, median, min, max, std)
    """
    if not values:
        return {}
    
    import statistics
    
    return {
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'min': min(values),
        'max': max(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0.0
    }