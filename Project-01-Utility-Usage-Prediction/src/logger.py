"""
Logger Module for Utility Usage Prediction Tool

This module provides centralized logging functionality for the application.
It configures logging to both file and console with appropriate formats.

Author: CodeVedX AI/ML Internship
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from config import LOG_FILE, APP_NAME


# ========================================
# LOG FORMATTERS
# ========================================

# Detailed formatter for file logging
FILE_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Simple formatter for console logging
CONSOLE_FORMATTER = logging.Formatter(
    fmt="%(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)


# ========================================
# LOGGER CLASS
# ========================================

class LoggerSetup:
    """
    Custom logger setup class for the application.
    
    This class provides a centralized logging configuration that logs
    to both file and console with different verbosity levels.
    """
    
    def __init__(self, name: str = APP_NAME):
        """
        Initialize the logger setup.
        
        Args:
            name: Logger name (defaults to APP_NAME from config)
        """
        self.name = name
        self.logger: Optional[logging.Logger] = None
    
    def get_logger(self) -> logging.Logger:
        """
        Get or create the application logger.
        
        Returns:
            Configured logger instance
        """
        if self.logger is None:
            self.logger = self._setup_logger()
        return self.logger
    
    def _setup_logger(self) -> logging.Logger:
        """
        Setup and configure the logger.
        
        Returns:
            Configured logger instance
        """
        # Create logger
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        
        # Prevent adding handlers multiple times
        if logger.handlers:
            return logger
        
        # Create file handler for detailed logging
        file_handler = self._create_file_handler()
        logger.addHandler(file_handler)
        
        # Create console handler for important messages
        console_handler = self._create_console_handler()
        logger.addHandler(console_handler)
        
        return logger
    
    def _create_file_handler(self) -> logging.FileHandler:
        """
        Create and configure file handler.
        
        Returns:
            Configured file handler
        """
        # Ensure log directory exists
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(
            filename=LOG_FILE,
            encoding="utf-8",
            mode="a"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(FILE_FORMATTER)
        
        return file_handler
    
    def _create_console_handler(self) -> logging.StreamHandler:
        """
        Create and configure console handler.
        
        Returns:
            Configured console handler
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(CONSOLE_FORMATTER)
        
        return console_handler


# ========================================
# GLOBAL LOGGER INSTANCE
# ========================================

# Create a global logger instance
_logger_setup = LoggerSetup()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get the application logger.
    
    This is the main function to get the logger instance throughout
    the application. Use this instead of creating new loggers.
    
    Args:
        name: Optional name for child logger (defaults to main logger)
    
    Returns:
        Logger instance
    
    Example:
        >>> from logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    if name:
        return logging.getLogger(f"{APP_NAME}.{name}")
    return _logger_setup.get_logger()


# ========================================
# CONVENIENCE LOGGING FUNCTIONS
# ========================================


def log_info(message: str) -> None:
    """Log info message."""
    get_logger().info(message)


def log_debug(message: str) -> None:
    """Log debug message."""
    get_logger().debug(message)


def log_warning(message: str) -> None:
    """Log warning message."""
    get_logger().warning(message)


def log_error(message: str, exc_info: bool = False) -> None:
    """
    Log error message.
    
    Args:
        message: Error message to log
        exc_info: Whether to include exception information
    """
    get_logger().error(message, exc_info=exc_info)


def log_critical(message: str, exc_info: bool = True) -> None:
    """
    Log critical error message.
    
    Args:
        message: Critical error message to log
        exc_info: Whether to include exception information
    """
    get_logger().critical(message, exc_info=exc_info)


def log_exception(message: str) -> None:
    """
    Log exception with traceback.
    
    Args:
        message: Exception message to log
    """
    get_logger().exception(message)


# ========================================
# APPLICATION EVENT LOGGERS
# ========================================


def log_application_start() -> None:
    """Log application startup event."""
    logger = get_logger()
    logger.info("=" * 80)
    logger.info(f"APPLICATION STARTED - {APP_NAME}")
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


def log_application_stop() -> None:
    """Log application shutdown event."""
    logger = get_logger()
    logger.info("=" * 80)
    logger.info(f"APPLICATION STOPPED - {APP_NAME}")
    logger.info(f"Stop Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


def log_dataset_loaded(file_path: str, rows: int, columns: int) -> None:
    """
    Log dataset loading event.
    
    Args:
        file_path: Path to the loaded dataset
        rows: Number of rows in the dataset
        columns: Number of columns in the dataset
    """
    logger = get_logger()
    logger.info(f"Dataset Loaded: {file_path}")
    logger.info(f"  - Rows: {rows}")
    logger.info(f"  - Columns: {columns}")


def log_record_added(record_id: int, details: str) -> None:
    """
    Log record addition event.
    
    Args:
        record_id: ID of the added record
        details: Details of the added record
    """
    logger = get_logger()
    logger.info(f"Record Added - ID: {record_id}")
    logger.debug(f"Details: {details}")


def log_record_updated(record_id: int, changes: str) -> None:
    """
    Log record update event.
    
    Args:
        record_id: ID of the updated record
        changes: Description of changes made
    """
    logger = get_logger()
    logger.info(f"Record Updated - ID: {record_id}")
    logger.debug(f"Changes: {changes}")


def log_record_deleted(record_id: int) -> None:
    """
    Log record deletion event.
    
    Args:
        record_id: ID of the deleted record
    """
    logger = get_logger()
    logger.info(f"Record Deleted - ID: {record_id}")


def log_model_trained(model_type: str, accuracy: float, duration: float) -> None:
    """
    Log model training event.
    
    Args:
        model_type: Type of model trained
        accuracy: Model accuracy/R² score
        duration: Training duration in seconds
    """
    logger = get_logger()
    logger.info(f"Model Trained - Type: {model_type}")
    logger.info(f"  - Accuracy (R² Score): {accuracy:.4f}")
    logger.info(f"  - Duration: {duration:.2f} seconds")


def log_prediction_made(input_features: dict, prediction: float) -> None:
    """
    Log prediction event.
    
    Args:
        input_features: Input features used for prediction
        prediction: Predicted value
    """
    logger = get_logger()
    logger.info("Prediction Made")
    logger.debug(f"Input Features: {input_features}")
    logger.info(f"Predicted Value: {prediction:.4f}")


def log_error_occurred(error_type: str, error_message: str, 
                       details: Optional[str] = None) -> None:
    """
    Log error event.
    
    Args:
        error_type: Type of error (e.g., 'ValueError', 'FileNotFoundError')
        error_message: Error message
        details: Additional error details
    """
    logger = get_logger()
    logger.error(f"Error Occurred - Type: {error_type}")
    logger.error(f"Message: {error_message}")
    if details:
        logger.error(f"Details: {details}")


# ========================================
# INITIALIZATION LOG
# ========================================

# Log module initialization
get_logger().debug("Logger module initialized successfully")