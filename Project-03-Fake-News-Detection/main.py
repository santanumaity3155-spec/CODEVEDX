"""
Main entry point for the Fake News Detection Tool.
This file serves as the entry point that calls the src/main.py module.
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from src.main import main
    
    if __name__ == "__main__":
        exit_code = main()
        sys.exit(exit_code)
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
