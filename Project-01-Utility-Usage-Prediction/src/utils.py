"""
Utility Functions Module for Utility Usage Prediction Tool

This module provides helper functions for console operations, formatting,
and common utilities used throughout the application.

Author: CodeVedX AI/ML Internship
"""

import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from config import APP_NAME, APP_VERSION, TABLE_WIDTH


# ========================================
# CONSOLE UTILITIES
# ========================================


def clear_console() -> None:
    """
    Clear the console screen based on the operating system.
    
    This function detects the OS and uses the appropriate command
    to clear the console screen.
    """
    os.system('cls' if os.name == 'nt' else 'clear')


def pause_program(message: str = "\nPress Enter to continue...") -> None:
    """
    Pause program execution and wait for user input.
    
    Args:
        message: Message to display before waiting for input
    """
    input(message)


def print_header(title: str, subtitle: Optional[str] = None) -> None:
    """
    Print a formatted header with title and optional subtitle.
    
    Args:
        title: Main title to display
        subtitle: Optional subtitle to display
    """
    print("\n" + "=" * TABLE_WIDTH)
    print(f"{title:^{TABLE_WIDTH}}")
    print("=" * TABLE_WIDTH)
    
    if subtitle:
        print(f"{subtitle:^{TABLE_WIDTH}}")
        print("-" * TABLE_WIDTH)


def print_separator(char: str = "=", length: Optional[int] = None) -> None:
    """
    Print a separator line.
    
    Args:
        char: Character to use for separator
        length: Length of separator (defaults to TABLE_WIDTH)
    """
    length = length or TABLE_WIDTH
    print(char * length)


def print_section(title: str) -> None:
    """
    Print a section header.
    
    Args:
        title: Section title to display
    """
    print(f"\n{title}")
    print("-" * len(title))


# ========================================
# TABLE FORMATTING UTILITIES
# ========================================


def print_table(
    headers: List[str],
    rows: List[List[Any]],
    title: Optional[str] = None,
    max_width: int = 20
) -> None:
    """
    Print data in a formatted table.
    
    Args:
        headers: List of column headers
        rows: List of rows (each row is a list of values)
        title: Optional table title
        max_width: Maximum width for each column
    """
    if not headers or not rows:
        print("No data to display")
        return
    
    # Calculate column widths
    col_widths = []
    for i, header in enumerate(headers):
        # Start with header length
        width = len(str(header))
        # Check all rows for this column
        for row in rows:
            if i < len(row):
                width = max(width, len(str(row[i])))
        # Limit to max_width
        col_widths.append(min(width, max_width))
    
    # Print title if provided
    if title:
        print(f"\n{title}")
        print_separator("-", TABLE_WIDTH)
    
    # Print header
    header_line = "|"
    for i, header in enumerate(headers):
        header_line += f" {str(header):^{col_widths[i]}} |"
    print(header_line)
    
    # Print separator
    sep_line = "|"
    for width in col_widths:
        sep_line += f"{'-' * (width + 2)}|"
    print(sep_line)
    
    # Print rows
    for row in rows:
        row_line = "|"
        for i, cell in enumerate(row):
            if i < len(col_widths):
                # Truncate if too long
                cell_str = str(cell)[:col_widths[i]]
                row_line += f" {cell_str:^{col_widths[i]}} |"
        print(row_line)
    
    print()


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format a number with specified decimal places.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
    
    Returns:
        Formatted number string
    """
    return f"{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a value as percentage.
    
    Args:
        value: Value to format (0-1 or 0-100)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    if value <= 1.0:
        value *= 100
    return f"{value:.{decimals}f}%"


# ========================================
# DATE AND TIME UTILITIES
# ========================================


def get_current_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get current date and time as formatted string.
    
    Args:
        format: Datetime format string
    
    Returns:
        Formatted datetime string
    """
    return datetime.now().strftime(format)


def get_current_date(format: str = "%Y-%m-%d") -> str:
    """
    Get current date as formatted string.
    
    Args:
        format: Date format string
    
    Returns:
        Formatted date string
    """
    return datetime.now().strftime(format)


def get_current_time(format: str = "%H:%M:%S") -> str:
    """
    Get current time as formatted string.
    
    Args:
        format: Time format string
    
    Returns:
        Formatted time string
    """
    return datetime.now().strftime(format)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted duration string (e.g., "2 hours 30 minutes 15 seconds")
    """
    delta = timedelta(seconds=seconds)
    
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if secs > 0 or not parts:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")
    
    return " ".join(parts)


def parse_date(date_string: str, format: str = "%Y-%m-%d") -> Optional[datetime]:
    """
    Parse date string to datetime object.
    
    Args:
        date_string: Date string to parse
        format: Expected date format
    
    Returns:
        Datetime object or None if parsing fails
    """
    try:
        return datetime.strptime(date_string, format)
    except ValueError:
        return None


# ========================================
# FILE UTILITIES
# ========================================


def ensure_directory(path: Path) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure
    """
    path.mkdir(parents=True, exist_ok=True)


def get_file_size(file_path: Path) -> str:
    """
    Get human-readable file size.
    
    Args:
        file_path: Path to file
    
    Returns:
        Formatted file size string (e.g., "2.5 MB")
    """
    if not file_path.exists():
        return "File not found"
    
    size_bytes = file_path.stat().st_size
    
    # Convert to appropriate unit
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} PB"


def list_files(directory: Path, extension: Optional[str] = None) -> List[Path]:
    """
    List files in a directory with optional extension filter.
    
    Args:
        directory: Directory path to list
        extension: Optional file extension filter (e.g., '.csv')
    
    Returns:
        List of file paths
    """
    if not directory.exists():
        return []
    
    files = [f for f in directory.iterdir() if f.is_file()]
    
    if extension:
        files = [f for f in files if f.suffix.lower() == extension.lower()]
    
    return sorted(files)


# ========================================
# DISPLAY UTILITIES
# ========================================


def print_success(message: str) -> None:
    """Print success message with checkmark."""
    print(f"✓ {message}")


def print_error(message: str) -> None:
    """Print error message with X mark."""
    print(f"✗ {message}")


def print_warning(message: str) -> None:
    """Print warning message with exclamation mark."""
    print(f"⚠ {message}")


def print_info(message: str) -> None:
    """Print info message with info icon."""
    print(f"ℹ {message}")


def print_banner(text: str, char: str = "=") -> None:
    """
    Print text in a banner format.
    
    Args:
        text: Text to display in banner
        char: Character to use for banner border
    """
    width = len(text) + 4
    print(char * width)
    print(f"{char} {text} {char}")
    print(char * width)


def center_text(text: str, width: Optional[int] = None) -> str:
    """
    Center text within specified width.
    
    Args:
        text: Text to center
        width: Width to center within (defaults to TABLE_WIDTH)
    
    Returns:
        Centered text string
    """
    width = width or TABLE_WIDTH
    return f"{text:^{width}}"


# ========================================
# DATA UTILITIES
# ========================================


def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate string to maximum length with suffix.
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_dict(data: Dict[str, Any], indent: int = 2) -> str:
    """
    Format dictionary for pretty printing.
    
    Args:
        data: Dictionary to format
        indent: Indentation level
    
    Returns:
        Formatted dictionary string
    """
    lines = ["{"]
    for key, value in data.items():
        lines.append(f"{' ' * indent}{key}: {value}")
    lines.append("}")
    return "\n".join(lines)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


# ========================================
# PROGRESS INDICATORS
# ========================================


def print_progress_bar(
    current: int,
    total: int,
    prefix: str = "Progress:",
    suffix: str = "Complete",
    length: int = 50,
    fill: str = "█"
) -> None:
    """
    Print a progress bar to console.
    
    Args:
        current: Current progress value
        total: Total value
        prefix: Prefix string
        suffix: Suffix string
        length: Character length of progress bar
        fill: Character to use for filled portion
    """
    percent = float(current) / float(total)
    filled_length = int(length * percent)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    print(f'\r{prefix} |{bar}| {percent:.1%} {suffix}', end='')
    
    if current == total:
        print()


def print_loading(message: str = "Loading...", delay: float = 0.5) -> None:
    """
    Print a loading message with animation.
    
    Args:
        message: Loading message
        delay: Delay between frames in seconds
    """
    import time
    
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    for frame in frames:
        print(f"\r{frame} {message}", end="", flush=True)
        time.sleep(delay)
    
    print()


# ========================================
# INPUT UTILITIES
# ========================================


def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """
    Get user input with optional default value.
    
    Args:
        prompt: Input prompt
        default: Default value if user enters empty string
    
    Returns:
        User input or default value
    """
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "
    
    user_input = input(full_prompt).strip()
    
    return user_input if user_input else (default or "")


def confirm_exit() -> bool:
    """
    Ask user to confirm exit.
    
    Returns:
        True if user confirms exit, False otherwise
    """
    response = input("\nAre you sure you want to exit? (y/n): ").strip().lower()
    return response in ['y', 'yes']


# ========================================
# APPLICATION INFO
# ========================================


def print_app_info() -> None:
    """Print application information banner."""
    print("\n" + "=" * TABLE_WIDTH)
    print(center_text(APP_NAME))
    print(center_text(f"Version {APP_VERSION}"))
    print(center_text("CodeVedX AI/ML Internship"))
    print("=" * TABLE_WIDTH)


def print_goodbye() -> None:
    """Print goodbye message."""
    print("\n" + "=" * TABLE_WIDTH)
    print(center_text("Thank you for using the application!"))
    print(center_text("Goodbye!"))
    print("=" * TABLE_WIDTH + "\n")


# ========================================
# ERROR HANDLING UTILITIES
# ========================================


def handle_keyboard_interrupt() -> None:
    """Handle keyboard interrupt (Ctrl+C) gracefully."""
    print("\n\n" + "=" * TABLE_WIDTH)
    print_warning("Operation interrupted by user")
    print("=" * TABLE_WIDTH)


def handle_unexpected_error(error: Exception) -> None:
    """
    Handle unexpected errors gracefully.
    
    Args:
        error: Exception that occurred
    """
    print("\n" + "=" * TABLE_WIDTH)
    print_error("An unexpected error occurred")
    print(f"Error Type: {type(error).__name__}")
    print(f"Error Message: {str(error)}")
    print("=" * TABLE_WIDTH)
    print("\nPlease check the logs for more details.")


# ========================================
# MISCELLANEOUS UTILITIES
# ========================================


def generate_timestamp() -> str:
    """
    Generate timestamp string for file naming.
    
    Returns:
        Timestamp string in format: YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_windows() -> bool:
    """Check if running on Windows."""
    return os.name == 'nt'


def is_linux() -> bool:
    """Check if running on Linux."""
    return os.name == 'posix' and sys.platform != 'darwin'


def is_mac() -> bool:
    """Check if running on macOS."""
    return sys.platform == 'darwin'


def get_platform() -> str:
    """
    Get current platform name.
    
    Returns:
        Platform name: 'Windows', 'Linux', or 'macOS'
    """
    if is_windows():
        return "Windows"
    elif is_mac():
        return "macOS"
    elif is_linux():
        return "Linux"
    else:
        return "Unknown"