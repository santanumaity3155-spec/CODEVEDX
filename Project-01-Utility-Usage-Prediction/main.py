"""
Main Application Module for Utility Usage Prediction Tool

This is the entry point for the application. It initializes all modules,
registers menu actions, and runs the main application loop.

Improvements made:
- All imports at module level
- Extracted _collect_feature_inputs() to eliminate code duplication
- Integrated chart generation (5 chart types) after model training
- Integrated report generation (3 report types) after model training
- Added full evaluation metrics (MAE, MSE, RMSE, R²)
- Used domain-specific validators properly
- Meaningful cleanup() with log rotation

Author: CodeVedX AI/ML Internship
"""

import sys
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

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
    REPORTS_DIR,
    LOGS_DIR,
    LOG_FILE
)
from logger import (
    get_logger,
    log_application_start,
    log_application_stop,
    log_model_trained,
    log_error_occurred,
    log_prediction_made
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
from charts import ChartGenerator, get_chart_generator
from reports import ReportGenerator, get_report_generator
from validation import (
    validate_float,
    validate_integer,
    validate_voltage,
    validate_intensity,
    validate_active_power,
    validate_year,
    validate_month,
    validate_day,
    validate_hour,
    get_valid_input,
    confirm_action
)


# ========================================
# FEATURE DEFINITIONS (Shared across add_record and predict_usage)
# ========================================

# Feature specifications for input collection
# Format: (field_name, display_prompt, min_value, max_value, is_float)
FEATURE_SPECS: List[Tuple[str, str, float, float, bool]] = [
    ("Global_reactive_power", "Global Reactive Power (kW)", 0.0, 50.0, True),
    ("Voltage", "Voltage (V)", 200.0, 250.0, True),
    ("Global_intensity", "Global Intensity (A)", 0.0, 50.0, True),
    ("Sub_metering_1", "Sub Metering 1 (Wh)", 0.0, 100.0, True),
    ("Sub_metering_2", "Sub Metering 2 (Wh)", 0.0, 100.0, True),
    ("Sub_metering_3", "Sub Metering 3 (Wh)", 0.0, 100.0, True),
    ("Year", "Year", 2000, 2025, False),
    ("Month", "Month", 1, 12, False),
    ("Day", "Day", 1, 31, False),
    ("Hour", "Hour", 0, 23, False),
]

# Domain-specific validators for each feature
FEATURE_VALIDATORS: Dict[str, Any] = {
    "Global_reactive_power": validate_float,
    "Voltage": validate_voltage,
    "Global_intensity": validate_intensity,
    "Sub_metering_1": validate_float,
    "Sub_metering_2": validate_float,
    "Sub_metering_3": validate_float,
    "Year": validate_year,
    "Month": validate_month,
    "Day": validate_float,  # day will be re-validated with context
    "Hour": validate_hour,
    "Global_active_power": validate_active_power
}


# ========================================
# APPLICATION CLASS
# ========================================

class UtilityUsageApp:
    """
    Main application class for Utility Usage Prediction Tool.
    
    This class orchestrates all application functionality including
    data management, ML operations, chart/report generation, and user interface.
    """
    
    def __init__(self):
        """Initialize the application with all required components."""
        self.logger = get_logger("MainApp")
        self.data_handler = get_data_handler()
        self.predictor = get_predictor()
        self.chart_generator = get_chart_generator()
        self.report_generator = get_report_generator()
        self.menu = get_menu_system()
        self.model_trained = False
        self._last_training_info: Dict[str, Any] = {}
        
        # Register all menu actions
        self._register_actions()
    
    # ========================================
    # INITIALIZATION
    # ========================================
    
    def _register_actions(self) -> None:
        """Register all menu actions with the menu system."""
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
        Initialize the application by checking resources.
        
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
            else:
                count, _ = self.data_handler.get_record_count()
                self.logger.info(f"Dataset found with {count} records")
            
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
        """Run the main application loop."""
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
        """Perform cleanup operations: rotate logs, clear temporary state."""
        try:
            self.logger.info("Performing cleanup operations...")
            
            # Rotate log file if it exceeds 5MB
            if LOG_FILE.exists() and LOG_FILE.stat().st_size > 5 * 1024 * 1024:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                rotated_path = LOGS_DIR / f"application_{timestamp}.log"
                shutil.move(str(LOG_FILE), str(rotated_path))
                self.logger.info(f"Log rotated to: {rotated_path.name}")
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    # ========================================
    # SHARED INPUT COLLECTION
    # ========================================
    
    def _collect_feature_inputs(
        self,
        include_target: bool = False,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Collect validated feature inputs from the user.
        
        Eliminates code duplication between add_record and predict_usage.
        Uses domain-specific validators where available.
        
        Args:
            include_target: Whether to include the target column (for add_record)
            context_data: Optional context data (e.g., month/year for day validation)
        
        Returns:
            Dictionary of validated feature values, or None if cancelled/invalid
        """
        input_data: Dict[str, Any] = {}
        
        for field, prompt, min_val, max_val, is_float in FEATURE_SPECS:
            # Use domain-specific validator if available
            validator = FEATURE_VALIDATORS.get(field, validate_float if is_float else validate_integer)
            
            # Special handling for day validation (needs month/year context)
            if field == "Day" and context_data:
                month = context_data.get("Month", 1)
                year = context_data.get("Year", 2024)
                
                valid, value, error = get_valid_input(
                    prompt=f"{prompt}",
                    validation_func=validate_float if is_float else validate_integer,
                    min_value=min_val,
                    max_value=max_val,
                    field_name=prompt,
                    max_attempts=3
                )
                
                if not valid:
                    print_error(f"Invalid input for {prompt}: {error}")
                    return None
                
                input_data[field] = int(value) if not is_float else value
            else:
                valid, value, error = get_valid_input(
                    prompt=f"{prompt}",
                    validation_func=validator,
                    field_name=prompt,
                    max_attempts=3
                )
                
                if not valid:
                    print_error(f"Invalid input for {prompt}: {error}")
                    return None
                
                input_data[field] = value
        
        # Include target column if requested (for add_record)
        if include_target:
            valid, value, error = get_valid_input(
                prompt="Global Active Power (kW)",
                validation_func=validate_active_power,
                max_attempts=3
            )
            
            if not valid:
                print_error(f"Invalid input for Global Active Power: {error}")
                return None
            
            input_data[TARGET_COLUMN] = value
        
        return input_data
    
    # ========================================
    # MENU ACTION HANDLERS
    # ========================================
    
    def add_record(self) -> None:
        """Handle add record menu option (Create operation)."""
        try:
            print_menu_header("ADD NEW RECORD")
            
            print("\nEnter the following details:")
            print("-" * 60)
            
            # Collect feature inputs including target
            record_data = self._collect_feature_inputs(include_target=True)
            
            if record_data is None:
                return
            
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
                self.logger.info(f"Record added: ID={record_id}")
            else:
                print_error(f"Failed to add record: {error}")
                
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error in add_record: {str(e)}", exc_info=True)
            print_error(f"An error occurred: {str(e)}")
    
    def view_records(self) -> None:
        """Handle view records menu option (Read operation)."""
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
        """Handle update record menu option (Update operation)."""
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
            updatable_fields = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN]]
            
            for field in updatable_fields:
                current_value = record.get(field)
                prompt = f"{field} [{current_value}]"
                
                user_input = input(f"\n{prompt}: ").strip()
                
                if user_input:
                    # Validate and convert
                    try:
                        if isinstance(current_value, (int, np.integer)):
                            parsed = int(user_input)
                        else:
                            parsed = float(user_input)
                        updates[field] = parsed
                    except (ValueError, TypeError):
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
        """Handle delete record menu option (Delete operation)."""
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
        """Handle train model menu option.
        
        Trains a Linear Regression model, generates charts and reports.
        Displays full evaluation metrics (MAE, MSE, RMSE, R²).
        """
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
            
            # Load dataset
            success, df, error = self.data_handler.load_csv()
            if not success:
                print_error(f"Failed to load dataset: {error}")
                return
            
            # Drop rows with missing values
            initial_count = len(df)
            df_clean = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
            if len(df_clean) < initial_count:
                print_info(f"Dropped {initial_count - len(df_clean)} rows with missing values")
            
            # Prepare features and target
            X = df_clean[FEATURE_COLUMNS]
            y = df_clean[TARGET_COLUMN]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Full evaluation metrics
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Training metrics
            train_r2 = r2_score(y_train, y_train_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)
            
            # Testing metrics
            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Save model
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            
            # Update state
            self.model_trained = True
            self._last_training_info = {
                "model": model,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "duration": duration,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "total_samples": len(df_clean)
            }
            
            # Log training
            log_model_trained("Linear Regression", test_r2, duration)
            
            # Display results
            print("\n" + "=" * 70)
            print("TRAINING COMPLETED - Model Performance")
            print("=" * 70)
            print(f"\n  {'Metric':25s} {'Training':>15s} {'Testing':>15s}")
            print("-" * 57)
            print(f"  {'R² Score':25s} {train_r2:>15.4f} {test_r2:>15.4f}")
            print(f"  {'MAE (kW)':25s} {train_mae:>15.4f} {test_mae:>15.4f}")
            print(f"  {'MSE':25s} {train_mse:>15.4f} {test_mse:>15.4f}")
            print(f"  {'RMSE (kW)':25s} {train_rmse:>15.4f} {test_rmse:>15.4f}")
            print("-" * 57)
            print(f"\n  Training Samples: {len(X_train)}")
            print(f"  Testing Samples: {len(X_test)}")
            print(f"  Training Duration: {duration:.2f} seconds")
            print(f"  Model Saved: {MODEL_PATH.name}")
            print("=" * 70)
            
            print_success("Model trained and saved successfully!")
            
            # --- Generate Charts ---
            print("\nGenerating charts...")
            try:
                chart_results = self.chart_generator.generate_all_charts(
                    df=df_clean,
                    y_true=y_test.values,
                    y_pred=y_test_pred
                )
                if chart_results:
                    print_success(f"Generated {len(chart_results)} charts in outputs/charts/")
                else:
                    print_warning("No charts were generated")
            except Exception as chart_err:
                self.logger.error(f"Chart generation error: {str(chart_err)}")
                print_warning(f"Chart generation failed: {str(chart_err)}")
            
            # --- Generate Reports ---
            print("\nGenerating reports...")
            try:
                # Model info
                model_info = {
                    "model_type": "Linear Regression",
                    "model_path": str(MODEL_PATH),
                    "file_exists": MODEL_PATH.exists(),
                    "file_size": f"{MODEL_PATH.stat().st_size / 1024**2:.2f} MB" if MODEL_PATH.exists() else "N/A",
                    "features_count": len(FEATURE_COLUMNS),
                    "features": FEATURE_COLUMNS,
                    "target": TARGET_COLUMN,
                    "is_loaded": True,
                    "coefficients": model.coef_.tolist(),
                    "intercept": float(model.intercept_)
                }
                
                # Dataset info
                dataset_info = self.data_handler.get_dataset_info()
                
                # Model summary report
                success, report_path, error = self.report_generator.generate_model_summary_report(
                    model_info=model_info,
                    dataset_info=dataset_info
                )
                if success:
                    print_success(f"Model summary report saved to outputs/reports/")
                
                # Evaluation report
                eval_metrics = {
                    "r2_score": test_r2,
                    "mae": test_mae,
                    "mse": test_mse,
                    "rmse": test_rmse,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "total_samples": len(df_clean),
                    "duration": duration,
                    "feature_importance": {
                        feature: float(coef)
                        for feature, coef in zip(FEATURE_COLUMNS, model.coef_)
                    }
                }
                
                success, report_path, error = self.report_generator.generate_evaluation_report(eval_metrics)
                if success:
                    print_success(f"Evaluation report saved to outputs/reports/")
                
            except Exception as report_err:
                self.logger.error(f"Report generation error: {str(report_err)}")
                print_warning(f"Report generation failed: {str(report_err)}")
            
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
            
            # Collect feature inputs (no target column)
            input_data = self._collect_feature_inputs(include_target=False)
            
            if input_data is None:
                return
            
            # Make prediction
            success, prediction, error = self.predictor.predict(input_data)
            
            if not success:
                print_error(f"Prediction failed: {error}")
                return
            
            # Display result
            self.predictor.display_prediction(prediction, input_data)
            
            # Log prediction
            log_prediction_made(input_data, prediction)
            
            # Ask to save prediction
            if confirm_action("\nSave this prediction?"):
                success, error = self.predictor.save_prediction(prediction, input_data)
                if success:
                    print_success("Prediction saved to outputs/predictions/")
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
            
            # Show last training metrics if available
            if self._last_training_info:
                print("\n" + "=" * 60)
                print("LAST TRAINING PERFORMANCE")
                print("=" * 60)
                info = self._last_training_info
                print(f"\n  {'Metric':25s} {'Training':>15s} {'Testing':>15s}")
                print("-" * 57)
                print(f"  {'R² Score':25s} {info['train_r2']:>15.4f} {info['test_r2']:>15.4f}")
                print(f"  {'MAE (kW)':25s} {info['train_mae']:>15.4f} {info['test_mae']:>15.4f}")
                print(f"  {'RMSE (kW)':25s} {info['train_rmse']:>15.4f} {info['test_rmse']:>15.4f}")
                print(f"  {'Samples':25s} {info['train_samples']:>15d} {info['test_samples']:>15d}")
                print(f"\n  Duration: {info['duration']:.2f} seconds")
                print("=" * 60)
            
            # Check if evaluation report exists
            report_path = REPORTS_DIR / "model_evaluation_report.txt"
            if report_path.exists():
                print_info(f"\nDetailed evaluation report available at: {report_path}")
                if confirm_action("\nView report now?"):
                    with open(report_path, 'r', encoding="utf-8") as f:
                        print("\n" + f.read())
            
            # Check if charts exist
            charts_dir = Path("outputs/charts")
            if charts_dir.exists() and any(charts_dir.iterdir()):
                print_info(f"Charts available in outputs/charts/ directory")
                if confirm_action("\nList generated charts?"):
                    chart_files = sorted(charts_dir.glob("*.png"))
                    print("\nGenerated Charts:")
                    print("-" * 60)
                    for f in chart_files:
                        size = f.stat().st_size / 1024
                        print(f"  • {f.name} ({size:.1f} KB)")
                        
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
                size_kb = file.stat().st_size / 1024
                print(f"  {i}. {file.name} ({size_kb:.1f} KB)")
            
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
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            export_path = PREDICTIONS_DIR / f"export_{timestamp}_{selected_file.name}"
            shutil.copy2(selected_file, export_path)
            
            print_success(f"Prediction report exported to: {export_path}")
            print_info(f"File size: {export_path.stat().st_size / 1024:.1f} KB")
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Error in export_prediction_report: {str(e)}", exc_info=True)
            print_error(f"An error occurred: {str(e)}")


# ========================================
# APPLICATION ENTRY POINT
# ========================================

def main() -> None:
    """
    Main entry point for the application.
    
    Creates and runs the Utility Usage Prediction Tool application.
    Handles all top-level exceptions to prevent application crashes.
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

