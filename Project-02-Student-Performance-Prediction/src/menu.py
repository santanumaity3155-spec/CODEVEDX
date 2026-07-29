"""
Menu Module
Provides the main menu interface and handles user interactions for all application features.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

from .config import MENU_OPTIONS, APP_NAME, APP_VERSION, MODEL_PATH
from .logger import app_logger, log_error, log_warning
from .utils import (
    print_header, print_subheader, print_separator, print_success,
    print_error, print_warning, print_info, print_table, clear_screen,
    pause_screen, get_user_confirmation
)
from .data_handler import DataHandler
from .predictor import Predictor
from .validation import (
    get_validated_input, validate_menu_choice, validate_search_query, 
    validate_string_input, validate_export_format, validate_file_path,
    validate_hours_studied, validate_attendance, validate_sleep_hours,
    validate_previous_scores, validate_tutoring_sessions, validate_physical_activity,
    validate_gender, validate_school_type, validate_teacher_quality,
    validate_parental_involvement, validate_access_to_resources,
    validate_extracurricular_activities, validate_motivation_level,
    validate_internet_access, validate_family_income, validate_peer_influence,
    validate_learning_disabilities, validate_parental_education_level,
    validate_distance_from_home
)


class Menu:
    """
    Main menu interface for the Student Performance Prediction System.
    
    Args:
        data_handler: DataHandler instance
        predictor: Predictor instance
    """
    def __init__(self, data_handler: DataHandler, predictor: Predictor):
        self.data_handler = data_handler
        self.predictor = predictor
        self.running = False
    
    def display_welcome(self) -> None:
        """Display welcome screen."""
        clear_screen()
        print_header(APP_NAME, char="=", length=80)
        print(f"\n{'Version: ' + APP_VERSION:^80}")
        print(f"{'Machine Learning-Powered Student Performance Prediction':^80}")
        print(f"\n{'Author: CodeVedX Intern':^80}")
        print_separator(char="=", length=80)
        print()
    
    def display_main_menu(self) -> None:
        """Display the main menu options."""
        print_header("MAIN MENU", char="-", length=80)
        
        for key, value in MENU_OPTIONS.items():
            print(f"{key:2d}. {value}")
        
        print_separator(char="-", length=80)
    
    def get_menu_choice(self) -> Optional[int]:
        """
        Get user's menu choice.
        
        Returns:
            Menu choice as integer or None if invalid
        """
        success, choice, error = get_validated_input(
            "\nEnter your choice (1-10): ",
            lambda x: validate_menu_choice(x, 1, 10),
            max_attempts=3
        )
        
        if success:
            return int(choice)
        else:
            print_error(f"Invalid input: {error}")
            return None
    
    def run(self) -> None:
        """Run the main application loop."""
        self.running = True
        
        app_logger.info("Starting main menu loop")
        
        while self.running:
            try:
                clear_screen()
                self.display_welcome()
                self.display_main_menu()
                
                choice = self.get_menu_choice()
                
                if choice is None:
                    print_warning("Too many invalid attempts. Please try again.")
                    pause_screen()
                    continue
                
                # Execute menu choice
                self.execute_choice(choice)
                
            except KeyboardInterrupt:
                print("\n")
                print_warning("Keyboard interrupt detected")
                if get_user_confirmation("Do you want to exit?"):
                    self.running = False
                else:
                    continue
            
            except Exception as e:
                log_error(f"Unexpected error in menu loop: {str(e)}", e)
                print_error(f"An unexpected error occurred: {str(e)}")
                pause_screen()
        
        self.shutdown()
    
    def execute_choice(self, choice: int) -> None:
        """
        Execute the selected menu option.
        
        Args:
            choice: Menu choice number
        """
        app_logger.info(f"User selected menu option: {choice}")
        
        menu_actions = {
            1: self.view_dataset_information,
            2: self.search_student,
            3: self.add_student_record,
            4: self.update_student_record,
            5: self.delete_student_record,
            6: self.predict_student_performance,
            7: self.batch_prediction,
            8: self.export_predictions,
            9: self.view_model_information,
            10: self.exit_application
        }
        
        action = menu_actions.get(choice)
        
        if action:
            action()
        else:
            print_error(f"Invalid choice: {choice}")
    
    def view_dataset_information(self) -> None:
        """Display dataset information (Menu Option 1)."""
        try:
            print_header("VIEW DATASET INFORMATION", char="=", length=80)
            
            # Display dataset summary
            self.data_handler.print_dataset_info()
            
            # Display sample records
            print_subheader("SAMPLE RECORDS (First 5)")
            
            headers, rows = self.data_handler.get_records_table_data()
            if rows:
                print_table(headers, rows[:5])
            else:
                print_info("No records in dataset")
            
            print_separator()
            pause_screen()
        
        except Exception as e:
            log_error("Error viewing dataset information", e)
            print_error(f"Error: {str(e)}")
            pause_screen()
    
    def search_student(self) -> None:
        """Search for student records (Menu Option 2)."""
        try:
            print_header("SEARCH STUDENT", char="=", length=80)
            
            # Get available search fields
            headers = self.data_handler.get_headers()
            
            print("\nAvailable search fields:")
            for i, header in enumerate(headers, 1):
                print(f"{i:2d}. {header}")
            
            print_separator(char="-", length=80)
            
            # Get search field
            field_success, field_choice, field_error = get_validated_input(
                "\nEnter field number to search: ",
                lambda x: validate_menu_choice(x, 1, len(headers)),
                max_attempts=3
            )
            
            if not field_success:
                print_error(f"Invalid field selection: {field_error}")
                pause_screen()
                return
            
            search_field = headers[int(field_choice) - 1]
            
            # Get search value
            value_success, search_value, value_error = get_validated_input(
                f"\nEnter search value for '{search_field}': ",
                validate_search_query,
                max_attempts=3
            )
            
            if not value_success:
                print_error(f"Invalid search value: {value_error}")
                pause_screen()
                return
            
            # Ask for match type
            print("\nSearch type:")
            print("1. Partial match (contains)")
            print("2. Exact match")
            
            match_success, match_choice, _ = get_validated_input(
                "\nEnter choice (1-2): ",
                lambda x: validate_menu_choice(x, 1, 2),
                max_attempts=3
            )
            
            exact_match = (int(match_choice) == 2) if match_success else False
            
            # Perform search
            print_info(f"Searching for '{search_value}' in '{search_field}'...")
            results = self.data_handler.search_records(search_field, search_value, exact_match)
            
            # Display results
            print_header("SEARCH RESULTS", char="-", length=80)
            
            if results:
                print(f"\nFound {len(results)} matching record(s)\n")
                
                headers, rows = self.data_handler.get_records_table_data(results)
                print_table(headers, rows)
            else:
                print_warning("No matching records found")
            
            print_separator()
            pause_screen()
        
        except Exception as e:
            log_error("Error searching student", e)
            print_error(f"Error: {str(e)}")
            pause_screen()
    
    def add_student_record(self) -> None:
        """Add a new student record (Menu Option 3)."""
        try:
            print_header("ADD STUDENT RECORD", char="=", length=80)
            
            print_info("Enter student information (press Enter to skip optional fields)")
            print_separator(char="-", length=80)
            
            # Collect student data
            record = {}
            
            print("\n--- Academic Information ---")
            
            # Hours Studied
            success, value, error = get_validated_input(
                "Hours Studied (0-50): ",
                validate_hours_studied,
                max_attempts=3
            )
            if success:
                record['Hours_Studied'] = value
            else:
                print_error("Failed to get valid input. Operation cancelled.")
                pause_screen()
                return
            
            # Attendance
            success, value, error = get_validated_input(
                "Attendance (0-100): ",
                validate_attendance,
                max_attempts=3
            )
            if success:
                record['Attendance'] = value
            else:
                print_error("Failed to get valid input. Operation cancelled.")
                pause_screen()
                return
            
            # Sleep Hours
            success, value, error = get_validated_input(
                "Sleep Hours (0-24): ",
                validate_sleep_hours,
                max_attempts=3
            )
            if success:
                record['Sleep_Hours'] = value
            else:
                print_error("Failed to get valid input. Operation cancelled.")
                pause_screen()
                return
            
            # Previous Scores
            success, value, error = get_validated_input(
                "Previous Scores (0-100): ",
                validate_previous_scores,
                max_attempts=3
            )
            if success:
                record['Previous_Scores'] = value
            else:
                print_error("Failed to get valid input. Operation cancelled.")
                pause_screen()
                return
            
            # Tutoring Sessions
            success, value, error = get_validated_input(
                "Tutoring Sessions (0-10): ",
                validate_tutoring_sessions,
                max_attempts=3
            )
            if success:
                record['Tutoring_Sessions'] = int(value)
            else:
                print_error("Failed to get valid input. Operation cancelled.")
                pause_screen()
                return
            
            # Physical Activity
            success, value, error = get_validated_input(
                "Physical Activity (0-10): ",
                validate_physical_activity,
                max_attempts=3
            )
            if success:
                record['Physical_Activity'] = int(value)
            else:
                print_error("Failed to get valid input. Operation cancelled.")
                pause_screen()
                return
            
            print("\n--- Personal Information ---")
            
            # Gender
            success, value, error = get_validated_input(
                "Gender (Male/Female): ",
                validate_gender,
                max_attempts=3
            )
            if success:
                record['Gender'] = value
            else:
                print_error("Failed to get valid input. Operation cancelled.")
                pause_screen()
                return
            
            # School Type
            success, value, error = get_validated_input(
                "School Type (Public/Private): ",
                validate_school_type,
                max_attempts=3
            )
            if success:
                record['School_Type'] = value
            else:
                print_error("Failed to get valid input. Operation cancelled.")
                pause_screen()
                return
            
            # Teacher Quality
            success, value, error = get_validated_input(
                "Teacher Quality (Low/Medium/High): ",
                validate_teacher_quality,
                max_attempts=3
            )
            if success:
                record['Teacher_Quality'] = value
            else:
                print_error("Failed to get valid input. Operation cancelled.")
                pause_screen()
                return
            
            # Optional fields with defaults
            print("\n--- Optional Information (press Enter for defaults) ---")
            
            optional_fields = {
                'Parental_Involvement': 'Low',
                'Access_to_Resources': 'Medium',
                'Extracurricular_Activities': 'No',
                'Motivation_Level': 'Medium',
                'Internet_Access': 'Yes',
                'Family_Income': 'Medium',
                'Peer_Influence': 'Neutral',
                'Learning_Disabilities': 'No',
                'Parental_Education_Level': 'High School',
                'Distance_from_Home': 'Near'
            }
            
            for field, default in optional_fields.items():
                value = input(f"{field.replace('_', ' ')} ({default}): ").strip()
                record[field] = value if value else default
            
            # Exam Score (optional for new records)
            exam_score = input("\nExam Score (0-100, press Enter to skip): ").strip()
            if exam_score:
                success, value, error = validate_previous_scores(exam_score)
                if success:
                    record['Exam_Score'] = value
                else:
                    print_warning(f"Invalid exam score: {error}. Skipping.")
            
            # Confirm addition
            print_separator(char="=", length=80)
            print("\nRecord Summary:")
            for key, value in record.items():
                print(f"  {key:30s}: {value}")
            
            print_separator(char="=", length=80)
            
            if not get_user_confirmation("\nDo you want to add this record?"):
                print_info("Operation cancelled")
                pause_screen()
                return
            
            # Add record
            success, message = self.data_handler.add_record(record)
            
            if success:
                print_success(message)
            else:
                print_error(message)
            
            pause_screen()
        
        except Exception as e:
            log_error("Error adding student record", e)
            print_error(f"Error: {str(e)}")
            pause_screen()
    
    def update_student_record(self) -> None:
        """Update an existing student record (Menu Option 4)."""
        try:
            print_header("UPDATE STUDENT RECORD", char="=", length=80)
            
            # First, search for the record
            print_info("Search for the record to update")
            print_separator(char="-", length=80)
            
            # Get search field
            headers = self.data_handler.get_headers()
            print("\nSearch by:")
            print("1. Record ID")
            print("2. Other field")
            
            search_success, search_choice, _ = get_validated_input(
                "\nEnter choice (1-2): ",
                lambda x: validate_menu_choice(x, 1, 2),
                max_attempts=3
            )
            
            if not search_success:
                print_error("Invalid choice")
                pause_screen()
                return
            
            results = []
            
            if int(search_choice) == 1:
                # Search by Record_ID
                success, record_id, error = get_validated_input(
                    "\nEnter Record ID: ",
                    validate_string_input,
                    max_attempts=3
                )
                
                if success:
                    results = self.data_handler.search_records('Record_ID', record_id, exact_match=True)
            else:
                # Search by other field
                print("\nAvailable fields:")
                for i, header in enumerate(headers, 1):
                    print(f"{i:2d}. {header}")
                
                field_success, field_choice, _ = get_validated_input(
                    "\nEnter field number: ",
                    lambda x: validate_menu_choice(x, 1, len(headers)),
                    max_attempts=3
                )
                
                if field_success:
                    search_field = headers[int(field_choice) - 1]
                    success, search_value, error = get_validated_input(
                        f"\nEnter search value for '{search_field}': ",
                        validate_string_input,
                        max_attempts=3
                    )
                    
                    if success:
                        results = self.data_handler.search_records(search_field, search_value)
            
            if not results:
                print_warning("No matching records found")
                pause_screen()
                return
            
            if len(results) > 1:
                print_warning(f"Multiple records found ({len(results)}). Please be more specific.")
                headers, rows = self.data_handler.get_records_table_data(results)
                print_table(headers, rows)
                pause_screen()
                return
            
            # Display record
            record = results[0]
            record_index = self.data_handler.records.index(record)
            
            print_header("RECORD TO UPDATE", char="-", length=80)
            headers, rows = self.data_handler.get_records_table_data([record])
            print_table(headers, rows)
            
            if not get_user_confirmation("\nDo you want to update this record?"):
                print_info("Operation cancelled")
                pause_screen()
                return
            
            # Get fields to update
            print("\nSelect fields to update:")
            for i, header in enumerate(headers, 1):
                print(f"{i:2d}. {header}")
            
            print("\nEnter field numbers to update (comma-separated, e.g., 1,3,5):")
            field_input = input("> ").strip()
            
            try:
                field_indices = [int(x.strip()) - 1 for x in field_input.split(',')]
                fields_to_update = [headers[i] for i in field_indices if 0 <= i < len(headers)]
            except (ValueError, IndexError):
                print_error("Invalid field selection")
                pause_screen()
                return
            
            if not fields_to_update:
                print_warning("No fields selected")
                pause_screen()
                return
            
            # Collect new values
            updated_data = {}
            
            for field in fields_to_update:
                print(f"\nUpdating: {field}")
                current_value = record.get(field, '')
                print(f"Current value: {current_value}")
                
                new_value = input(f"New value (press Enter to keep current): ").strip()
                
                if new_value:
                    updated_data[field] = new_value
            
            if not updated_data:
                print_info("No changes made")
                pause_screen()
                return
            
            # Confirm update
            print_separator(char="=", length=80)
            print("\nUpdate Summary:")
            for field, value in updated_data.items():
                print(f"  {field:30s}: {record.get(field)} -> {value}")
            
            print_separator(char="=", length=80)
            
            if not get_user_confirmation("\nConfirm update?"):
                print_info("Operation cancelled")
                pause_screen()
                return
            
            # Update record
            success, message = self.data_handler.update_record(record_index, updated_data)
            
            if success:
                print_success(message)
            else:
                print_error(message)
            
            pause_screen()
        
        except Exception as e:
            log_error("Error updating student record", e)
            print_error(f"Error: {str(e)}")
            pause_screen()
    
    def delete_student_record(self) -> None:
        """Delete a student record (Menu Option 5)."""
        try:
            print_header("DELETE STUDENT RECORD", char="=", length=80)
            
            # Search for record
            print_info("Search for the record to delete")
            print_separator(char="-", length=80)
            
            success, search_value, error = get_validated_input(
                "\nEnter Record ID or search term: ",
                validate_string_input,
                max_attempts=3
            )
            
            if not success:
                print_error(f"Invalid input: {error}")
                pause_screen()
                return
            
            # Search in multiple fields
            results = self.data_handler.search_by_multiple_fields({
                'Record_ID': search_value,
                'Gender': search_value,
                'School_Type': search_value
            })
            
            if not results:
                print_warning("No matching records found")
                pause_screen()
                return
            
            # Display results
            print_header("MATCHING RECORDS", char="-", length=80)
            headers, rows = self.data_handler.get_records_table_data(results)
            print_table(headers, rows)
            
            if len(results) > 1:
                print(f"\n{len(results)} records found. Please be more specific.")
                pause_screen()
                return
            
            # Confirm deletion
            record = results[0]
            record_index = self.data_handler.records.index(record)
            
            print("\nRecord to delete:")
            for key, value in record.items():
                print(f"  {key:30s}: {value}")
            
            print_separator(char="=", length=80)
            
            if not get_user_confirmation("\nAre you sure you want to DELETE this record?"):
                print_info("Operation cancelled")
                pause_screen()
                return
            
            # Delete record
            success, message = self.data_handler.delete_record(record_index)
            
            if success:
                print_success(message)
            else:
                print_error(message)
            
            pause_screen()
        
        except Exception as e:
            log_error("Error deleting student record", e)
            print_error(f"Error: {str(e)}")
            pause_screen()
    
    def predict_student_performance(self) -> None:
        """Predict student performance for a single student (Menu Option 6)."""
        try:
            print_header("PREDICT STUDENT PERFORMANCE", char="=", length=80)
            
            # Check if model is loaded
            if not self.predictor.is_model_ready():
                print_error("Model is not loaded. Cannot make predictions.")
                print_info(f"Expected model path: {MODEL_PATH}")
                if not self.predictor.is_model_loaded():
                    print_error("Model file not found or failed to load.")
                    print_info("Please ensure the model file exists and is valid.")
                pause_screen()
                return
            
            print_info("Enter student information for prediction")
            print_separator(char="-", length=80)
            
            input_data = {}
            
            print("\n--- Required Information ---")
            
            # Hours Studied
            success, value, error = get_validated_input(
                "Hours Studied (0-50): ",
                validate_hours_studied,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Hours_Studied'] = value
            
            # Attendance
            success, value, error = get_validated_input(
                "Attendance (0-100): ",
                validate_attendance,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Attendance'] = value
            
            # Sleep Hours
            success, value, error = get_validated_input(
                "Sleep Hours (0-24): ",
                validate_sleep_hours,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Sleep_Hours'] = value
            
            # Previous Scores
            success, value, error = get_validated_input(
                "Previous Scores (0-100): ",
                validate_previous_scores,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Previous_Scores'] = value
            
            # Tutoring Sessions
            success, value, error = get_validated_input(
                "Tutoring Sessions (0-10): ",
                validate_tutoring_sessions,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Tutoring_Sessions'] = int(value)
            
            # Physical Activity
            success, value, error = get_validated_input(
                "Physical Activity (0-10): ",
                validate_physical_activity,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Physical_Activity'] = int(value)
            
            print("\n--- Personal Information ---")
            
            # Gender
            success, value, error = get_validated_input(
                "Gender (Male/Female): ",
                validate_gender,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Gender'] = value
            
            # School Type
            success, value, error = get_validated_input(
                "School Type (Public/Private): ",
                validate_school_type,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['School_Type'] = value
            
            # Teacher Quality
            success, value, error = get_validated_input(
                "Teacher Quality (Low/Medium/High): ",
                validate_teacher_quality,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Teacher_Quality'] = value
            
            print("\n--- Family & Social Information ---")
            
            # Parental Involvement
            success, value, error = get_validated_input(
                "Parental Involvement (Low/Medium/High): ",
                validate_parental_involvement,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Parental_Involvement'] = value
            
            # Access to Resources
            success, value, error = get_validated_input(
                "Access to Resources (Low/Medium/High): ",
                validate_access_to_resources,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Access_to_Resources'] = value
            
            # Extracurricular Activities
            success, value, error = get_validated_input(
                "Extracurricular Activities (Yes/No): ",
                validate_extracurricular_activities,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Extracurricular_Activities'] = value
            
            # Motivation Level
            success, value, error = get_validated_input(
                "Motivation Level (Low/Medium/High): ",
                validate_motivation_level,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Motivation_Level'] = value
            
            # Internet Access
            success, value, error = get_validated_input(
                "Internet Access (Yes/No): ",
                validate_internet_access,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Internet_Access'] = value
            
            # Family Income
            success, value, error = get_validated_input(
                "Family Income (Low/Medium/High): ",
                validate_family_income,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Family_Income'] = value
            
            # Peer Influence
            success, value, error = get_validated_input(
                "Peer Influence (Negative/Neutral/Positive): ",
                validate_peer_influence,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Peer_Influence'] = value
            
            # Learning Disabilities
            success, value, error = get_validated_input(
                "Learning Disabilities (Yes/No): ",
                validate_learning_disabilities,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Learning_Disabilities'] = value
            
            # Parental Education Level
            success, value, error = get_validated_input(
                "Parental Education Level (High School/College/Postgraduate): ",
                validate_parental_education_level,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Parental_Education_Level'] = value
            
            # Distance from Home
            success, value, error = get_validated_input(
                "Distance from Home (Near/Moderate/Far): ",
                validate_distance_from_home,
                max_attempts=3
            )
            if not success:
                print_error("Invalid input. Operation cancelled.")
                pause_screen()
                return
            input_data['Distance_from_Home'] = value
            
            # Make prediction
            print_info("\nMaking prediction...")
            success, prediction, message = self.predictor.predict_single(input_data)
            
            if success and prediction is not None:
                self.predictor.display_prediction_result(prediction, input_data)
                
                # Ask if user wants to save prediction
                if get_user_confirmation("\nDo you want to save this prediction?"):
                    self.predictor.save_predictions([{
                        'index': 1,
                        'input': input_data,
                        'prediction': prediction,
                        'success': True,
                        'message': 'Success'
                    }])
            else:
                print_error(f"Prediction failed: {message}")
            
            pause_screen()
        
        except Exception as e:
            log_error("Error predicting student performance", e)
            print_error(f"Error: {str(e)}")
            pause_screen()
    
    def batch_prediction(self) -> None:
        """Perform batch prediction from CSV file (Menu Option 7)."""
        try:
            print_header("BATCH PREDICTION", char="=", length=80)
            
            # Check if model is loaded
            if not self.predictor.is_model_ready():
                print_error("Model is not loaded. Cannot make predictions.")
                print_info(f"Expected model path: {MODEL_PATH}")
                if not self.predictor.is_model_loaded():
                    print_error("Model file not found or failed to load.")
                    print_info("Please ensure the model file exists and is valid.")
                pause_screen()
                return
            
            print_info("Batch prediction allows you to predict performance for multiple students")
            print_info("using a CSV file containing student data")
            print_separator(char="-", length=80)
            
            # Get CSV file path
            success, file_path, error = get_validated_input(
                "\nEnter path to CSV file: ",
                lambda x: validate_file_path(x, must_exist=True),
                max_attempts=3
            )
            
            if not success:
                print_error(f"Invalid file path: {error}")
                pause_screen()
                return
            
            csv_path = Path(file_path)
            
            # Confirm operation
            print(f"\nCSV file: {csv_path}")
            if not get_user_confirmation("\nProceed with batch prediction?"):
                print_info("Operation cancelled")
                pause_screen()
                return
            
            # Make predictions
            print_info("\nProcessing batch prediction...")
            success, results, message = self.predictor.predict_from_csv(csv_path)
            
            if success:
                print_success(message)
                
                # Display results
                self.predictor.display_batch_results(results)
                
                # Ask if user wants to save results
                if get_user_confirmation("\nDo you want to save these predictions?"):
                    self.predictor.save_predictions(results)
            else:
                print_error(f"Batch prediction failed: {message}")
            
            pause_screen()
        
        except Exception as e:
            log_error("Error in batch prediction", e)
            print_error(f"Error: {str(e)}")
            pause_screen()
    
    def export_predictions(self) -> None:
        """Export predictions to file (Menu Option 8)."""
        try:
            print_header("EXPORT PREDICTIONS", char="=", length=80)
            
            print_info("This feature allows you to export dataset or predictions to a file")
            print_separator(char="-", length=80)
            
            print("\nExport options:")
            print("1. Export current dataset")
            print("2. Export predictions from CSV")
            
            success, choice, _ = get_validated_input(
                "\nEnter choice (1-2): ",
                lambda x: validate_menu_choice(x, 1, 2),
                max_attempts=3
            )
            
            if not success:
                print_error("Invalid choice")
                pause_screen()
                return
            
            if int(choice) == 1:
                # Export dataset
                format_success, export_format, format_error = get_validated_input(
                    "\nEnter export format (csv/json): ",
                    validate_export_format,
                    max_attempts=3
                )
                
                if not format_success:
                    print_error(f"Invalid format: {format_error}")
                    pause_screen()
                    return
                
                from .utils import get_filename_timestamp
                from .config import OUTPUTS_DIR
                output_path = OUTPUTS_DIR / f"exports/dataset_export_{get_filename_timestamp()}.{export_format}"
                
                success, message = self.data_handler.export_dataset(output_path, export_format)
                
                if success:
                    print_success(message)
                else:
                    print_error(message)
            
            else:
                # Export predictions
                file_success, csv_path, file_error = get_validated_input(
                    "\nEnter path to predictions CSV file: ",
                    lambda x: validate_file_path(x, must_exist=True),
                    max_attempts=3
                )
                
                if not file_success:
                    print_error(f"Invalid file path: {file_error}")
                    pause_screen()
                    return
                
                from .utils import get_filename_timestamp
                from .config import OUTPUTS_DIR
                output_path = OUTPUTS_DIR / f"exports/predictions_export_{get_filename_timestamp()}.csv"
                
                # Read and export
                from .utils import read_csv_file, export_to_csv
                records, headers = read_csv_file(Path(csv_path))
                
                if export_to_csv(records, output_path, headers):
                    print_success(f"Predictions exported to: {output_path}")
                else:
                    print_error("Failed to export predictions")
            
            pause_screen()
        
        except Exception as e:
            log_error("Error exporting predictions", e)
            print_error(f"Error: {str(e)}")
            pause_screen()
    
    def view_model_information(self) -> None:
        """Display model information (Menu Option 9)."""
        try:
            self.predictor.print_model_info()
            pause_screen()
        
        except Exception as e:
            log_error("Error viewing model information", e)
            print_error(f"Error: {str(e)}")
            pause_screen()
    
    def exit_application(self) -> None:
        """Exit the application (Menu Option 10)."""
        print_header("EXIT APPLICATION", char="=", length=80)
        
        if get_user_confirmation("\nAre you sure you want to exit?"):
            self.running = False
            print_info("Thank you for using the application!")
        else:
            print_info("Exit cancelled")
            pause_screen()
    
    def shutdown(self) -> None:
        """Perform cleanup operations before shutdown."""
        from .logger import log_shutdown
        
        print("\n")
        print_info("Shutting down application...")
        
        # Log shutdown
        log_shutdown()
        
        print_success("Application closed successfully")