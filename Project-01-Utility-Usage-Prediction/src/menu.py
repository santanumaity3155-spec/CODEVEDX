"""
Menu Module for Utility Usage Prediction Tool

This module provides a professional console-based menu system for the application.
It handles menu display, user input, and navigation between different features.

Author: CodeVedX AI/ML Internship
"""

from typing import Callable, Dict, Any, Optional
from pathlib import Path

from config import MENU_OPTIONS, APP_NAME, APP_VERSION
from utils import (
    clear_console,
    print_header,
    print_section,
    print_success,
    print_error,
    print_info,
    print_warning,
    print_app_info,
    print_goodbye,
    handle_keyboard_interrupt
)
from logger import get_logger


# ========================================
# MENU SYSTEM CLASS
# ========================================

class MenuSystem:
    """
    Professional console menu system for the application.
    
    This class provides a structured menu interface with proper formatting,
    input validation, and navigation handling.
    """
    
    def __init__(self):
        """Initialize the menu system."""
        self.logger = get_logger("MenuSystem")
        self.running = False
        self.menu_actions: Dict[int, Callable] = {}
    
    # ========================================
    # MENU DISPLAY
    # ========================================
    
    def display_main_menu(self) -> None:
        """Display the main application menu."""
        clear_console()
        print_app_info()
        
        print("\n" + "=" * 60)
        print("MAIN MENU")
        print("=" * 60)
        
        for option_num, option_text in MENU_OPTIONS.items():
            print(f"  {option_num:2d}. {option_text}")
        
        print("=" * 60)
    
    def display_welcome_screen(self) -> None:
        """Display welcome screen with application information."""
        clear_console()
        
        welcome_text = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              UTILITY USAGE PREDICTION TOOL                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Welcome to the Utility Usage Prediction Tool!

This application helps you:
  • Manage utility usage records
  • Train machine learning models
  • Predict future utility consumption
  • Analyze energy usage patterns

Features:
  ✓ Add, view, update, and delete records
  ✓ Train Linear Regression models
  ✓ Make predictions on new data
  ✓ Export predictions and reports

Developed for: CodeVedX AI/ML Internship
Version: 1.0.0
        """
        
        print(welcome_text)
    
    def display_goodbye_screen(self) -> None:
        """Display goodbye screen."""
        clear_console()
        print_goodbye()
    
    # ========================================
    # MENU NAVIGATION
    # ========================================
    
    def get_menu_choice(self, min_choice: int = 1, max_choice: int = 10) -> Optional[int]:
        """
        Get user's menu choice with validation.
        
        Args:
            min_choice: Minimum valid menu option
            max_choice: Maximum valid menu option
        
        Returns:
            User's choice or None if invalid
        """
        try:
            choice_input = input(f"\nEnter your choice ({min_choice}-{max_choice}): ").strip()
            
            # Validate input
            if not choice_input:
                print_warning("Please enter a choice")
                return None
            
            choice = int(choice_input)
            
            if choice < min_choice or choice > max_choice:
                print_error(f"Invalid choice. Please enter a number between {min_choice} and {max_choice}")
                return None
            
            return choice
            
        except ValueError:
            print_error("Invalid input. Please enter a valid number")
            return None
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error getting menu choice: {str(e)}")
            print_error(f"An error occurred: {str(e)}")
            return None
    
    def confirm_action(self, message: str = "Are you sure?") -> bool:
        """
        Ask user to confirm an action.
        
        Args:
            message: Confirmation message
        
        Returns:
            True if user confirms, False otherwise
        """
        try:
            response = input(f"\n{message} (y/n): ").strip().lower()
            
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print_warning("Please enter 'y' or 'n'")
                return self.confirm_action(message)
                
        except KeyboardInterrupt:
            return False
        except Exception as e:
            self.logger.error(f"Error in confirmation: {str(e)}")
            return False
    
    def pause(self, message: str = "\nPress Enter to continue...") -> None:
        """
        Pause execution and wait for user input.
        
        Args:
            message: Message to display
        """
        try:
            input(message)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error in pause: {str(e)}")
    
    # ========================================
    # ACTION REGISTRATION
    # ========================================
    
    def register_action(self, option_number: int, action: Callable) -> None:
        """
        Register an action for a menu option.
        
        Args:
            option_number: Menu option number
            action: Function to call when option is selected
        """
        self.menu_actions[option_number] = action
        self.logger.debug(f"Registered action for menu option {option_number}")
    
    def register_actions(self, actions: Dict[int, Callable]) -> None:
        """
        Register multiple actions at once.
        
        Args:
            actions: Dictionary mapping option numbers to actions
        """
        for option_number, action in actions.items():
            self.register_action(option_number, action)
    
    # ========================================
    # MENU EXECUTION
    # ========================================
    
    def execute_action(self, choice: int) -> bool:
        """
        Execute the action for the selected menu option.
        
        Args:
            choice: User's menu choice
        
        Returns:
            True to continue, False to exit
        """
        try:
            # Check if action is registered
            if choice not in self.menu_actions:
                print_error(f"No action registered for option {choice}")
                return True
            
            # Get the action
            action = self.menu_actions[choice]
            
            # Execute the action
            self.logger.info(f"Executing menu option {choice}: {MENU_OPTIONS.get(choice, 'Unknown')}")
            result = action()
            
            # Action can return False to exit
            if result is False:
                return False
            
            return True
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error executing menu action: {str(e)}", exc_info=True)
            print_error(f"An error occurred: {str(e)}")
            return True
    
    def run(self, welcome_message: bool = True) -> None:
        """
        Run the main menu loop.
        
        Args:
            welcome_message: Whether to show welcome message
        """
        self.running = True
        
        # Show welcome message
        if welcome_message:
            self.display_welcome_screen()
            self.pause("\nPress Enter to continue to main menu...")
        
        # Main menu loop
        while self.running:
            try:
                # Display menu
                self.display_main_menu()
                
                # Get user choice
                choice = self.get_menu_choice()
                
                if choice is None:
                    self.pause("\nPress Enter to continue...")
                    continue
                
                # Handle exit choice
                if choice == 10:  # Exit
                    if self.confirm_action("Are you sure you want to exit?"):
                        break
                    continue
                
                # Execute action
                continue_running = self.execute_action(choice)
                
                if not continue_running:
                    break
                
                # Pause before showing menu again
                if choice != 10:
                    self.pause("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt detected")
                print("\n")
                if self.confirm_action("Do you want to exit?"):
                    break
            except Exception as e:
                self.logger.error(f"Unexpected error in menu loop: {str(e)}", exc_info=True)
                print_error(f"An unexpected error occurred: {str(e)}")
                self.pause("\nPress Enter to continue...")
        
        # Show goodbye message
        self.display_goodbye_screen()
    
    def stop(self) -> None:
        """Stop the menu loop."""
        self.running = False
        self.logger.info("Menu system stopped")


# ========================================
# MENU HEADER AND FOOTER
# ========================================


def print_menu_header(title: str) -> None:
    """
    Print a formatted menu section header.
    
    Args:
        title: Section title
    """
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_menu_footer() -> None:
    """Print a formatted menu section footer."""
    print("=" * 60)


def print_submenu(title: str, options: Dict[int, str]) -> None:
    """
    Print a submenu with options.
    
    Args:
        title: Submenu title
        options: Dictionary mapping option numbers to descriptions
    """
    print(f"\n{title}")
    print("-" * len(title))
    
    for option_num, option_text in options.items():
        print(f"  {option_num:2d}. {option_text}")


# ========================================
# INPUT PROMPTS
# ========================================


def get_input_with_prompt(prompt: str, required: bool = True) -> Optional[str]:
    """
    Get user input with a prompt.
    
    Args:
        prompt: Input prompt
        required: Whether input is required
    
    Returns:
        User input or None if cancelled
    """
    try:
        while True:
            user_input = input(f"\n{prompt}: ").strip()
            
            if not user_input and required:
                print_warning("This field is required. Please enter a value.")
                continue
            
            return user_input if user_input else None
            
    except KeyboardInterrupt:
        return None
    except Exception as e:
        print_error(f"Error getting input: {str(e)}")
        return None


def get_yes_no_input(prompt: str, default: bool = False) -> bool:
    """
    Get yes/no input from user.
    
    Args:
        prompt: Question to ask
        default: Default value if user enters empty string
    
    Returns:
        True for yes, False for no
    """
    try:
        default_str = "y/n"
        if default:
            default_str = "Y/n"
        else:
            default_str = "y/N"
        
        while True:
            response = input(f"\n{prompt} [{default_str}]: ").strip().lower()
            
            if not response:
                return default
            
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print_warning("Please enter 'y' or 'n'")
    
    except KeyboardInterrupt:
        return default
    except Exception as e:
        print_error(f"Error getting input: {str(e)}")
        return default


# ========================================
# MENU UTILITIES
# ========================================


def display_section_title(title: str) -> None:
    """
    Display a section title with formatting.
    
    Args:
        title: Section title
    """
    print("\n" + "-" * 60)
    print(f"  {title}")
    print("-" * 60)


def display_success_message(message: str) -> None:
    """
    Display a success message.
    
    Args:
        message: Success message
    """
    print_success(message)


def display_error_message(message: str) -> None:
    """
    Display an error message.
    
    Args:
        message: Error message
    """
    print_error(message)


def display_warning_message(message: str) -> None:
    """
    Display a warning message.
    
    Args:
        message: Warning message
    """
    print_warning(message)


def display_info_message(message: str) -> None:
    """
    Display an info message.
    
    Args:
        message: Info message
    """
    print_info(message)


# ========================================
# GLOBAL MENU INSTANCE
# ========================================

# Create a global menu instance
_menu_system = MenuSystem()


def get_menu_system() -> MenuSystem:
    """
    Get the global menu system instance.
    
    Returns:
        MenuSystem instance
    """
    return _menu_system


def run_menu(welcome_message: bool = True) -> None:
    """
    Run the menu system (convenience function).
    
    Args:
        welcome_message: Whether to show welcome message
    """
    menu = get_menu_system()
    menu.run(welcome_message=welcome_message)