"""
Main entry point for the Fake News Detection Tool.
Initializes the application and starts the menu system.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.menu import show_menu
from src.logger import logger


def main():
    """
    Main function to start the application.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        logger.info("Starting Fake News Detection Tool...")
        show_menu()
        return 0
    except KeyboardInterrupt:
        print("\n\n⚠️  Application interrupted by user")
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.critical(f"Fatal error in main: {str(e)}", exc_info=True)
        print(f"\n❌ Fatal error: {str(e)}")
        print("Please check logs/application.log for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)