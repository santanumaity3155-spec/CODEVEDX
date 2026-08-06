"""
Logging configuration for the Fake News Detection Tool.
Sets up logging to both console and file with appropriate formatting.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

from src.config import LOG_DIR, LOG_FILE


def setup_logger():
    """
    Configure and return the application logger.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("fake_news_detection")
    logger.setLevel(logging.INFO)
    
    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_formatter = logging.Formatter(
        "%(levelname)s: %(message)s"
    )
    
    # File handler - logs everything to file
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler - logs INFO and above to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log startup
    logger.info("=" * 70)
    logger.info("FAKE NEWS DETECTION TOOL - APPLICATION STARTUP")
    logger.info(f"Startup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log File: {LOG_FILE}")
    logger.info("=" * 70)
    
    return logger


# Create and export the logger instance
logger = setup_logger()