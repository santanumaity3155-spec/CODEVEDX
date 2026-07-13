"""
Main Application Module for Utility Usage Prediction Tool

This is the entry point for the application. It initializes all modules,
registers menu actions, and runs the main application loop.

Author: CodeVedX AI/ML Internship
"""

import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (
    APP_NAME,
    APP_VERSION,
    DATASET_PATH,
    MODEL_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    PREDICTIONS_DIR,
    REPORTS_DIR
)
from logger import (
    get_logger,
    log_application_start,
    log_application_stop,
    log_model_trained,
    log_error_occurred
)
from utils import (
    clear_console,
    print_header,
    print_section,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_app_info,
    format_number,
    handle_unexpected_error
)
from data_handler import DataHandler, get_data_handler
from predictor import MLPredictor, get_predictor
from menu import MenuSystem, get_menu_system, print_menu_header, print_menu_footer
from validation import (
    validate_float,
    validate_integer,
    validate_string,
    get_valid_input,
    confirm_action
)


# ========================================
# APPLICATION CLASS
# ========================================

class UtilityUsageApp:
    """
    Main application class for Utility Usage Prediction Tool.
    
    This class orchestrates all application functionality including
    data management, ML operations, and user interface.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.logger = get_logger("MainApp")
        self.data_handler = get_data_handler()
        self.predictor = get_predictor()
        self.menu = get_menu_system()
        self.model_trained = False
        
        # Register all menu actions
        self._register_actions()
    
    # ========================================
    # INITIALIZATION
    # ========================================
    
    def _register_actions(self) -> None:
        """Register all menu actions."""
        actions = {
            1: self.add_record,
            2: self.view_records,
            3: self.search_record,
            4: self.update_record,
            5: self.delete_record,
            6: self.train_model,
            7: self.predict_usage,
            8: self.view_model_metrics,
            9: self.export_prediction_report
        }
        
        self.menu.register_actions(actions)
        self.logger.info("All menu actions registered")
    
    def initialize(self) -> bool:
        """
        Initialize the application.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info(f"Initializing {APP_NAME} v{APP_VERSION}")
            self.logger.info("=" * 80)
            
            # Check if dataset exists
            if not DATASET_PATH.exists():
                self.logger.warning(f"Dataset not found at {DATASET_PATH}")
                print_warning(f"Dataset not found at {DATASET_PATH}")
                print_info("A new dataset will be created when you add records")
            
            # Check if model exists
            if MODEL_PATH.exists():
                self.logger.info(f"Pre-trained model found at {MODEL_PATH}")
                print_info(f"Pre-trained model found at {MODEL_PATH.name}")
                self.model_trained = True
            else:
                self.logger.info("No pre-trained model found. Model training required.")
                print_info("No pre-trained model found. Please train the model first (Option 6)")
            
            self.logger.info("Application initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize application: {str(e)}", exc_info=True)
            print_error(f"Failed to initialize application: {str(e)}")
            return False
    
    def run(self) -> None:
        """Run the main application."""
        try:
            # Log application start
            log_application_start()
            
            # Initialize application
            if not self.initialize():
                print_error("Application initialization failed. Exiting...")
                return
            
            # Run menu loop
            self.menu.run(welcome_message=True)
            
            # Log application stop
            log_application_stop()
            
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
            print("\n")
            print_warning("Application interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error in application: {str(e)}", exc_info=True)
            handle_unexpected_error(e)
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Perform cleanup operations."""
        try:
            self.logger.info("Performing cleanup operations...")
            # Add any cleanup code here if needed
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    # ========================================
    # MENU ACTION HANDLERS
    # ========================================
    
    def add_record(self) -> None:
        """Handle add record menu option."""
        try:
            print_menu_header("ADD NEW RECORD")
            
            print("\nEnter the following details:")
            print("-" * 60)
            
            # Collect record data
            record_data = {}
            
            # Get numeric features
            features = {
                "Global_reactive_power": ("Global Reactive Power (kW)", 0.0, 50.0),
                "Voltage": ("Voltage (V)", 200.0, 250.0),
                "Global_intensity": ("Global Intensity (A)", 0.0, 50.0),
                "Sub_metering_1": ("Sub Metering 1 (Wh)", 0.0, 100.0),
                "Sub_metering_2": ("Sub Metering 2 (Wh)", 0.0, 100.0),
                "Sub_metering_3": ("Sub Metering 3 (Wh)", 0.0, 100.0),
                "Year": ("Year", 2000, 2024),
                "Month": ("Month", 1, 12),
                "Day": ("Day", 1, 31),
                "Hour": ("Hour", 0, 23),
                "Global_active_power": ("Global Active Power (kW)", 0.0, 10.0)
            }
            
            for feature, (prompt, min_val, max_val) in features.items():
                valid, value, error = get_valid_input(
                    prompt=f"{prompt}",
                    validation_func=validate_float if isinstance(min_val, float) else validate_integer,
                    min_value=min_val,
                    max_value=max_val,
                    field_name=prompt,
                    max_attempts=3
                )
                
                if not valid:
                    print_error(f"Failed to get valid input for {prompt}: {error}")
                    return
                
                record_data[feature] = value
            
            # Confirm addition
            print("\nRecord to be added:")
            print("-" * 60)
            for key, value in record_data.items():
                print(f"  {key:25s}: {value}")
            
            if not confirm_action("\nAdd this record?"):
                print_info("Record addition cancelled")
                return
            
            # Add record
            success, record_id, error = self.data_handler.add_record(record_data)
            
            if success:
                print_success(f"Record added successfully with ID: {record_id}")
            else:
                print_error(f"Failed to add record: {error}")
                
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error in add_record: {str(e)}", exc_info=True)
            print_error(f"An error occurred: {str(e)}")
    
    def view_records(self) -> None:
        """Handle view records menu option."""
        try:
            print_menu_header("VIEW RECORDS")
            
            # Get all records
            success, df, error = self.data_handler.get_all_records(limit=100)
            
            if not success:
                print_error(f"Failed to load records: {error}")
                return
            
            if df is None or df.empty:
                print_info("No records found in the dataset")
                return
            
            # Display records
            self.data_handler.display_records(df, max_records=100)
            
            # Show count
            count, _ = self.data_handler.get_record_count()
            print_info(f"Total records: {count}")
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error in view_records: {str(e)}", exc_info=True)
            print_error(f"An error occurred: {str(e)}")
    
    def search_record(self) -> None:
        """Handle search record menu option."""
        try:
            print_menu_header("SEARCH RECORD")
            
            # Get existing IDs
            existing_ids = self.data_handler.get_existing_ids()
            
            if not existing_ids:
                print_info("No records found in the dataset")
                return
            
            # Get record ID
            valid, record_id, error = get_valid_input(
                prompt="Enter Record ID",
                validation_func=validate_integer,
                min_value=1,
                field_name="Record ID",
                max_attempts=3
            )
            
            if not valid:
                print_error(f"Invalid input: {error}")
                return
            
            # Search for record
            success, record, error = self.data_handler.search_record(record_id)
            
            if not success:
                print_error(f"Record not found: {error}")
                return
            
            # Display record details
            self.data_handler.display_record_details(record)
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error in search_record: {str(e)}", exc_info=True)
            print_error(f"An error occurred: {str(e)}")
    
    def update_record(self) -> None:
        """Handle update record menu option."""
        try:
            print_menu_header("UPDATE RECORD")
            
            # Get existing IDs
            existing_ids = self.data_handler.get_existing_ids()
            
            if not existing_ids:
                print_info("No records found in the dataset")
                return
            
            # Get record ID
            valid, record_id, error = get_valid_input(
                prompt="Enter Record ID to update",
                validation_func=validate_integer,
                min_value=1,
                field_name="Record ID",
                max_attempts=3
            )
            
            if not valid:
                print_error(f"Invalid input: {error}")
                return
            
            # Check if record exists
            if not self.data_handler.record_exists(record_id):
                print_error(f"Record with ID {record_id} not found")
                return
            
            # Get current record
            success, record, error = self.data_handler.search_record(record_id)
            if not success:
                print_error(f"Failed to retrieve record: {error}")
                return
            
            # Display current record
            print("\nCurrent Record:")
            print("-" * 60)
            self.data_handler.display_record_details(record)
            
            # Get fields to update
            print("\nEnter new values (press Enter to keep current value):")
            print("-" * 60)
            
            updates = {}
            updatable_fields = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col != 'ID']
            
            for field in updatable_fields:
                current_value = record.get(field)
                prompt = f"{field} [{current_value}]"
                
                user_input = input(f"\n{prompt}: ").strip()
                
                if user_input:
                    # Validate and convert
                    try:
                        if isinstance(current_value, int):
                            updates[field] = int(user_input)
                        else:
                            updates[field] = float(user_input)
                    except ValueError:
                        print_warning(f"Invalid input for {field}, keeping current value")
            
            if not updates:
                print_info("No changes made")
                return
            
            # Confirm update
            print("\nChanges to be made:")
            print("-" * 60)
            for key, value in updates.items():
                print(f"  {key:25s}: {record.get(key)} -> {value}")
            
            if not confirm_action("\nUpdate this record?"):
                print_info("Update cancelled")
                return
            
            # Update record
            success, error = self.data_handler.update_record(record_id, updates)
            
            if success:
                print_success(f"Record {record_id} updated successfully")
            else:
                print_error(f"Failed to update record: {error}")
                
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error in update_record: {str(e)}", exc_info=True)
            print_error(f"An error occurred: {str(e)}")
    
    def delete_record(self) -> None:
        """Handle delete record menu option."""
        try:
            print_menu_header("DELETE RECORD")
            
            # Get existing IDs
            existing_ids = self.data_handler.get_existing_ids()
            
            if not existing_ids:
                print_info("No records found in the dataset")
                return
            
            # Get record ID
            valid, record_id, error = get_valid_input(
                prompt="Enter Record ID to delete",
                validation_func=validate_integer,
                min_value=1,
                field_name="Record ID",
                max_attempts=3
            )
            
            if not valid:
                print_error(f"Invalid input: {error}")
                return
            
            # Get record details
            success, record, error = self.data_handler.search_record(record_id)
            if not success:
                print_error(f"Record not found: {error}")
                return
            
            # Display record
            print("\nRecord to be deleted:")
            print("-" * 60)
            self.data_handler.display_record_details(record)
            
            # Confirm deletion
            if not confirm_action("\nAre you sure you want to delete this record?"):
                print_info("Deletion cancelled")
                return
            
            # Delete record
            success, error = self.data_handler.delete_record(record_id)
            
            if success:
                print_success(f"Record {record_id} deleted successfully")
            else:
                print_error(f"Failed to delete record: {error}")
                
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error in delete_record: {str(e)}", exc_info=True)
            print_error(f"An error occurred: {str(e)}")
    
    def train_model(self) -> None:
        """Handle train model menu option."""
        try:
            print_menu_header("TRAIN ML MODEL")
            
            # Check if dataset exists
            if not DATASET_PATH.exists():
                print_error("Dataset not found. Please add records first.")
                return
            
            # Check if dataset has records
            count, _ = self.data_handler.get_record_count()
            if count < 10:
                print_warning(f"Dataset has only {count} records. Minimum 10 records recommended for training.")
                if not confirm_action("Continue anyway?"):
                    return
            
            print("\nTraining Configuration:")
            print("-" * 60)
            print(f"  Dataset: {DATASET_PATH.name}")
            print(f"  Records: {count}")
            print(f"  Model Type: Linear Regression")
            print(f"  Target: {TARGET_COLUMN}")
            print(f"  Features: {len(FEATURE_COLUMNS)}")
            print("-" * 60)
            
            if not confirm_action("\nStart training?"):
                print_info("Training cancelled")
                return
            
            print("\nTraining model... This may take a few moments.")
            
            # Record start time
            start_time = time.time()
            
            # Import training function
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            import pandas as pd
            
            # Load dataset
            success, df, error = self.data_handler.load_csv()
            if not success:
                print_error(f"Failed to load dataset: {error}")
                return
            
            # Prepare features and target
            X = df[FEATURE_COLUMNS]
            y = df[TARGET_COLUMN]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Save model
            import joblib
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            
            # Update state
            self.model_trained = True
            
            # Log training
            log_model_trained("Linear Regression", r2, duration)
            
            # Display results
            print("\n" + "=" * 60)
            print("TRAINING COMPLETED")
            print("=" * 60)
            print(f"\n  Model Type: Linear Regression")
            print(f"  Training Samples: {len(X_train)}")
            print(f"  Testing Samples: {len(X_test)}")
            print(f"  R² Score: {r2:.4f}")
            print(f"  Training Duration: {duration:.2f} seconds")
            print(f"  Model Saved: {MODEL_PATH.name}")
            print("=" * 60)
            
            print_success("Model trained and saved successfully!")
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error in train_model: {str(e)}", exc_info=True)
            print_error(f"Training failed: {str(e)}")
    
    def predict_usage(self) -> None:
        """Handle predict usage menu option."""
        try:
            print_menu_header("PREDICT UTILITY USAGE")
            
            # Check if model is trained
            if not self.model_trained and not MODEL_PATH.exists():
                print_error("Model not found. Please train the model first (Option 6)")
                return
            
            # Load model if not loaded
            if not self.predictor.is_model_loaded():
                success, error = self.predictor.load_model()
                if not success:
                    print_error(f"Failed to load model: {error}")
                    return
            
            print("\nEnter the following features for prediction:")
            print("-" * 60)
            
            # Collect input data
            input_data = {}
            
            features = {
                "Global_reactive_power": ("Global Reactive Power (kW)", 0.0, 50.0),
                "Voltage": ("Voltage (V)", 200.0, 250.0),
                "Global_intensity": ("Global Intensity (A)", 0.0, 50.0),
                "Sub_metering_1": ("Sub Metering 1 (Wh)", 0.0, 100.0),
                "Sub_metering_2": ("Sub Metering 2 (Wh)", 0.0, 100.0),
                "Sub_metering_3": ("Sub Metering 3 (Wh)", 0.0, 100.0),
                "Year": ("Year", 2000, 2024),
                "Month": ("Month", 1, 12),
                "Day": ("Day", 1, 31),
                "Hour": ("Hour", 0, 23)
            }
            
            for feature, (prompt, min_val, max_val) in features.items():
                valid, value, error = get_valid_input(
                    prompt=f"{prompt}",
                    validation_func=validate_float if isinstance(min_val, float) else validate_integer,
                    min_value=min_val,
                    max_value=max_val,
                    field_name=prompt,
                    max_attempts=3
                )
                
                if not valid:
                    print_error(f"Invalid input for {prompt}: {error}")
                    return
                
                input_data[feature] = value
            
            # Make prediction
            success, prediction, error = self.predictor.predict(input_data)
            
            if not success:
                print_error(f"Prediction failed: {error}")
                return
            
            # Display result
            self.predictor.display_prediction(prediction, input_data)
            
            # Ask to save prediction
            if confirm_action("\nSave this prediction?"):
                success, error = self.predictor.save_prediction(prediction, input_data)
                if success:
                    print_success("Prediction saved successfully")
                else:
                    print_error(f"Failed to save prediction: {error}")
                    
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error in predict_usage: {str(e)}", exc_info=True)
            print_error(f"An error occurred: {str(e)}")
    
    def view_model_metrics(self) -> None:
        """Handle view model metrics menu option."""
        try:
            print_menu_header("MODEL METRICS")
            
            # Check if model exists
            if not MODEL_PATH.exists():
                print_error("Model not found. Please train the model first (Option 6)")
                return
            
            # Load model if not loaded
            if not self.predictor.is_model_loaded():
                success, error = self.predictor.load_model()
                if not success:
                    print_error(f"Failed to load model: {error}")
                    return
            
            # Display model information
            self.predictor.display_model_metrics()
            
            # Check if evaluation report exists
            report_path = REPORTS_DIR / "model_evaluation_report.txt"
            if report_path.exists():
                print_info(f"\nDetailed evaluation report available at: {report_path}")
                if confirm_action("\nView report now?"):
                    with open(report_path, 'r') as f:
                        print("\n" + f.read())
                        
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error in view_model_metrics: {str(e)}", exc_info=True)
            print_error(f"An error occurred: {str(e)}")
    
    def export_prediction_report(self) -> None:
        """Handle export prediction report menu option."""
        try:
            print_menu_header("EXPORT PREDICTION REPORT")
            
            # Check if predictions exist
            if not PREDICTIONS_DIR.exists() or not any(PREDICTIONS_DIR.iterdir()):
                print_info("No predictions found. Make some predictions first (Option 7)")
                return
            
            # List available prediction files
            prediction_files = sorted(PREDICTIONS_DIR.glob("*.csv"), reverse=True)
            
            if not prediction_files:
                print_info("No prediction files found")
                return
            
            print("\nAvailable prediction files:")
            print("-" * 60)
            for i, file in enumerate(prediction_files[:10], 1):
                size = file.stat().st_size
                print(f"  {i}. {file.name} ({size} bytes)")
            
            # Get user choice
            valid, choice, error = get_valid_input(
                prompt="\nEnter file number to export (0 to cancel)",
                validation_func=validate_integer,
                min_value=0,
                max_value=len(prediction_files[:10]),
                field_name="Choice",
                max_attempts=3
            )
            
            if not valid or choice == 0:
                print_info("Export cancelled")
                return
            
            # Get selected file
            selected_file = prediction_files[choice - 1]
            
            # Export file
            import shutil
            export_path = PREDICTIONS_DIR / f"export_{selected_file.name}"
            shutil.copy2(selected_file, export_path)
            
            print_success(f"Prediction report exported to: {export_path}")
            print_info(f"File size: {export_path.stat().st_size} bytes")
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error in export_prediction_report: {str(e)}", exc_info=True)
            print_error(f"An error occurred: {str(e)}")


# ========================================
# APPLICATION ENTRY POINT
# ========================================

def main():
    """
    Main entry point for the application.
    
    This function creates and runs the Utility Usage Prediction Tool application.
    """
    try:
        # Create application instance
        app = UtilityUsageApp()
        
        # Run the application
        app.run()
        
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print("Please check the logs for more details.")
        sys.exit(1)


if __name__ == "__main__":
    main()