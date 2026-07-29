"""
Main Application Module
Entry point for the Student Performance Prediction System.
Initializes components and starts the application.
"""

import sys
import signal
from pathlib import Path

from .config import APP_NAME, APP_VERSION, PROJECT_ROOT, MODEL_PATH
from .logger import app_logger, log_startup, log_shutdown, log_error
from .utils import print_success, print_error, print_info, print_warning
from .data_handler import DataHandler
from .predictor import Predictor
from .menu import Menu


def signal_handler(sig, frame):
    """
    Handle keyboard interrupt (Ctrl+C) gracefully.
    
    Args:
        sig: Signal number
        frame: Current stack frame
    """
    print("\n")
    print("Keyboard interrupt detected. Shutting down gracefully...")
    log_shutdown()
    sys.exit(0)


def check_dependencies() -> bool:
    """
    Check if required dependencies are available.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    required_packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'pickle'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print_error(f"Missing required packages: {', '.join(missing_packages)}")
        print_info("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    return True


def initialize_application() -> tuple[DataHandler, Predictor, bool]:
    """
    Initialize application components.
    
    Returns:
        Tuple of (DataHandler, Predictor, success)
    """
    from .utils import print_header
    
    # Initialize DataHandler
    print_header("INITIALIZING DATA HANDLER", char="=", length=80)
    data_handler = DataHandler()
    
    if data_handler.get_record_count() == 0:
        print_warning("Dataset is empty. You can add records using the menu.")
    else:
        print_success(f"Dataset Loaded: {data_handler.get_record_count()} records")
    
    # Initialize Predictor
    print_header("INITIALIZING PREDICTOR", char="=", length=80)
    predictor = Predictor()
    
    if predictor.is_model_ready():
        print_success("Model Loaded")
    else:
        print_error("Warning: Model could not be loaded. Prediction features will be unavailable.")
        print_info(f"Expected model path: {MODEL_PATH}")
        print_info("Please ensure the model file exists and is valid.")
    
    # Initialize directories
    from .config import create_directories
    create_directories()
    print_success("Directories Initialized")
    
    # Logger status
    print_success("Logger Ready")
    
    return data_handler, predictor, True


def print_banner() -> None:
    """Print application banner."""
    print("\n" + "=" * 80)
    print(f"{APP_NAME:^80}")
    print(f"{'Version ' + APP_VERSION:^80}")
    print(f"{'Machine Learning-Powered Student Performance Prediction':^80}")
    print("=" * 80)
    print()


def main():
    """
    Main entry point for the application.
    """
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Print banner
        print_banner()
        
        # Log startup
        log_startup()
        
        # Check dependencies
        print_info("Checking dependencies...")
        if not check_dependencies():
            log_error("Dependency check failed")
            sys.exit(1)
        
        print_success("All dependencies available")
        
        # Initialize application
        print_info("Initializing application components...")
        data_handler, predictor, success = initialize_application()
        
        if not success:
            log_error("Application initialization failed")
            print_error("Failed to initialize application")
            sys.exit(1)
        
        print_success("Application initialized successfully")
        
        # Create and run menu
        menu = Menu(data_handler, predictor)
        
        print_success("Starting application...")
        
        # Run the application
        menu.run()
        
    except KeyboardInterrupt:
        print("\n")
        print_warning("Application interrupted by user")
        log_shutdown()
        sys.exit(0)
    
    except Exception as e:
        log_error(f"Fatal error in main: {str(e)}", e)
        print_error(f"Fatal error: {str(e)}")
        print_info("Please check the logs for more details")
        sys.exit(1)


if __name__ == "__main__":
    main()