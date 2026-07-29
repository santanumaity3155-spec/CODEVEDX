"""
Student Performance Prediction System
A machine learning-powered console application for predicting student exam performance.
"""

__version__ = "1.0.0"
__author__ = "CodeVedX Intern"
__description__ = "Machine Learning-Powered Student Performance Prediction System"

from .config import APP_NAME, APP_VERSION
from .logger import app_logger
from .data_handler import DataHandler
from .predictor import Predictor
from .menu import Menu

__all__ = [
    'APP_NAME',
    'APP_VERSION',
    'app_logger',
    'DataHandler',
    'Predictor',
    'Menu'
]