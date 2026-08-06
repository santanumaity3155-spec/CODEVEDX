"""
Utility functions for the Fake News Detection Tool.
Provides helper functions for UI, formatting, and common operations.
"""

import os
import sys
import platform
import psutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import PROJECT_ROOT


def clear_screen():
    """Clear the console screen based on the operating system."""
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
    except Exception:
        pass


def pause(message: str = "\nPress Enter to continue..."):
    """
    Pause execution and wait for user input.
    
    Args:
        message: Message to display before waiting
    """
    input(message)


def print_banner(title: str = "", subtitle: str = ""):
    """
    Print a professional ASCII banner.
    
    Args:
        title: Main title text
        subtitle: Optional subtitle text
    """
    width = 70
    print("\n" + "=" * width)
    if title:
        print(title.center(width))
    if subtitle:
        print(subtitle.center(width))
    print("=" * width + "\n")


def print_section_header(title: str):
    """
    Print a section header with ASCII borders.
    
    Args:
        title: Section title text
    """
    width = 70
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)


def format_number(num: int) -> str:
    """
    Format a number with commas for readability.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted number string
    """
    return f"{num:,}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a float as a percentage string.
    
    Args:
        value: Float value (0-1 or 0-100)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    if value <= 1.0:
        value = value * 100
    return f"{value:.{decimals}f}%"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted file size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_file_size(file_path: Path) -> str:
    """
    Get formatted file size.
    
    Args:
        file_path: Path to file
        
    Returns:
        Formatted file size string or error message
    """
    try:
        if file_path.exists():
            size = file_path.stat().st_size
            return format_file_size(size)
        return "File not found"
    except Exception as e:
        return f"Error: {str(e)}"


def validate_file_exists(file_path: Path) -> tuple[bool, str]:
    """
    Check if a file exists and is accessible.
    
    Args:
        file_path: Path to check
        
    Returns:
        Tuple of (exists: bool, message: str)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"
    try:
        with open(file_path, 'r') as f:
            pass
        return True, "File is accessible"
    except PermissionError:
        return False, f"Permission denied: {file_path}"
    except Exception as e:
        return False, f"Error accessing file: {str(e)}"


def get_memory_usage() -> dict:
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory usage details
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
            "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
            "percent": round(process.memory_percent(), 2)
        }
    except Exception:
        return {"rss_mb": 0, "vms_mb": 0, "percent": 0}


def get_system_info() -> dict:
    """
    Get system information.
    
    Returns:
        Dictionary with system details
    """
    try:
        return {
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "working_directory": str(PROJECT_ROOT),
        }
    except Exception as e:
        return {"error": str(e)}


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Format a timestamp for display.
    
    Args:
        timestamp: datetime object (uses current time if None)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def calculate_processing_time(start_time: datetime, end_time: datetime) -> str:
    """
    Calculate and format processing time.
    
    Args:
        start_time: Start datetime
        end_time: End datetime
        
    Returns:
        Formatted processing time string
    """
    delta = end_time - start_time
    total_seconds = delta.total_seconds()
    
    if total_seconds < 1:
        return f"{total_seconds * 1000:.2f} ms"
    elif total_seconds < 60:
        return f"{total_seconds:.2f} seconds"
    else:
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes}m {seconds:.2f}s"


def confirm_action(prompt: str = "Are you sure?") -> bool:
    """
    Ask user for confirmation.
    
    Args:
        prompt: Confirmation prompt text
        
    Returns:
        True if user confirms, False otherwise
    """
    while True:
        response = input(f"\n{prompt} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        print("Please enter 'y' or 'n'.")


def get_user_input(prompt: str, required: bool = True) -> str:
    """
    Get user input with optional validation.
    
    Args:
        prompt: Input prompt text
        required: Whether input is required
        
    Returns:
        User input string
    """
    while True:
        value = input(prompt).strip()
        if value or not required:
            return value
        print("This field is required. Please enter a value.")


def display_progress_bar(current: int, total: int, prefix: str = "Progress", 
                        suffix: str = "Complete", length: int = 50):
    """
    Display a progress bar in the console.
    
    Args:
        current: Current progress value
        total: Total value
        prefix: Prefix text
        suffix: Suffix text
        length: Bar length in characters
    """
    percent = (current / total) * 100
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='')
    if current == total:
        print()