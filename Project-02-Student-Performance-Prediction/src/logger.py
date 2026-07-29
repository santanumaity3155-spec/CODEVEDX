"""
Logger Module
Configures and provides logging functionality for the application.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from .config import LOG_FILE, LOGS_DIR


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to console output.
    """
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            record.levelname = colored_levelname
        
        return super().format(record)


def setup_logger(
    name: str = "student_performance_app",
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up and configure the application logger.
    
    Args:
        name: Logger name
        log_file: Path to log file (defaults to config.LOG_FILE)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    if log_file is None:
        log_file = LOG_FILE
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = "student_performance_app") -> logging.Logger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name (use module name for better tracking)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Application-wide logger instance
app_logger = setup_logger()


def log_startup():
    """Log application startup."""
    app_logger.info("=" * 60)
    app_logger.info("Application Started")
    app_logger.info(f"Log file: {LOG_FILE}")
    app_logger.info("=" * 60)


def log_shutdown():
    """Log application shutdown."""
    app_logger.info("=" * 60)
    app_logger.info("Application Shutting Down")
    app_logger.info("=" * 60)


def log_csv_loaded(file_path: Path, records_count: int):
    """
    Log CSV file loading.
    
    Args:
        file_path: Path to the loaded CSV file
        records_count: Number of records loaded
    """
    app_logger.info(f"CSV Loaded: {file_path} | Records: {records_count}")


def log_record_operation(operation: str, record_id: str, success: bool = True):
    """
    Log record operations (add, update, delete).
    
    Args:
        operation: Operation type (Added, Updated, Deleted)
        record_id: Record identifier
        success: Whether operation was successful
    """
    status = "SUCCESS" if success else "FAILED"
    app_logger.info(f"Record {operation}: {record_id} | Status: {status}")


def log_prediction(input_data: dict, prediction: float, success: bool = True):
    """
    Log prediction operations.
    
    Args:
        input_data: Input features used for prediction
        prediction: Predicted value
        success: Whether prediction was successful
    """
    status = "SUCCESS" if success else "FAILED"
    app_logger.info(f"Prediction {status} | Input: {input_data} | Predicted Score: {prediction:.2f}")


def log_error(message: str, exception: Optional[Exception] = None):
    """
    Log error messages.
    
    Args:
        message: Error message
        exception: Exception object (optional)
    """
    if exception:
        app_logger.error(f"{message} | Exception: {str(exception)}", exc_info=True)
    else:
        app_logger.error(message)


def log_warning(message: str):
    """
    Log warning messages.
    
    Args:
        message: Warning message
    """
    app_logger.warning(message)


def log_debug(message: str):
    """
    Log debug messages.
    
    Args:
        message: Debug message
    """
    app_logger.debug(message)