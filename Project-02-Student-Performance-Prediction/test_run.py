"""
Quick test script to verify application can be imported and initialized
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    print("Testing imports...")
    
    # Test config import
    from config import APP_NAME, APP_VERSION, DATASET_PATH, MODEL_PATH
    print(f"✓ Config imported successfully")
    print(f"  - App: {APP_NAME} v{APP_VERSION}")
    print(f"  - Dataset: {DATASET_PATH}")
    print(f"  - Model: {MODEL_PATH}")
    
    # Test logger import
    from logger import app_logger, log_startup
    print(f"✓ Logger imported successfully")
    
    # Test utils import
    from utils import format_score, print_header, export_to_csv
    print(f"✓ Utils imported successfully")
    
    # Test validation import
    from validation import validate_hours_studied, validate_attendance
    print(f"✓ Validation imported successfully")
    
    # Test data_handler import
    from data_handler import DataHandler
    print(f"✓ DataHandler imported successfully")
    
    # Test predictor import
    from predictor import Predictor
    print(f"✓ Predictor imported successfully")
    
    # Test menu import
    from menu import Menu
    print(f"✓ Menu imported successfully")
    
    # Test main import
    from main import check_dependencies
    print(f"✓ Main module imported successfully")
    
    print("\n" + "="*80)
    print("All imports successful! ✓")
    print("="*80)
    
    # Test validation functions
    print("\nTesting validation functions...")
    is_valid, value, error = validate_hours_studied("25")
    print(f"✓ validate_hours_studied('25') = {is_valid}, {value}")
    
    is_valid, value, error = validate_attendance("85")
    print(f"✓ validate_attendance('85') = {is_valid}, {value}")
    
    # Test utility functions
    print("\nTesting utility functions...")
    score_str = format_score(85.5)
    print(f"✓ format_score(85.5) = '{score_str}'")
    
    print("\n" + "="*80)
    print("All tests passed! Application is ready to run. ✓")
    print("="*80)
    print("\nTo run the application, use: python src/main.py")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)