"""
Student Performance Prediction System
Root-level entry point for the application.

This file allows running the application using: python main.py
instead of: python src/main.py
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now import and run the application
try:
    # Import the src.main module directly
    from src import main as src_main
    src_main.main()
    
except ImportError as e:
    print(f"Error importing application modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Fatal error: {e}")
    sys.exit(1)
