"""
Data Handler Module for Utility Usage Prediction Tool

This module provides comprehensive data management functionality including:
- Loading and saving CSV files
- Adding, updating, and deleting records
- Searching and displaying records
- Automatic ID generation

Author: CodeVedX AI/ML Internship
"""

import csv
import os
import shutil
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import pandas as pd

from config import (
    DATASET_PATH,
    CSV_DELIMITER,
    CSV_ENCODING,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)
from logger import get_logger, log_dataset_loaded, log_record_added, log_record_updated, log_record_deleted
from utils import print_success, print_error, print_info, print_table, format_number


# ========================================
# DATA HANDLER CLASS
# ========================================

class DataHandler:
    """
    Data handler class for managing utility usage records.
    
    This class provides methods to perform CRUD operations on the dataset
    and manage CSV file operations.
    """
    
    def __init__(self, dataset_path: Path = DATASET_PATH):
        """
        Initialize the data handler.
        
        Args:
            dataset_path: Path to the dataset CSV file
        """
        self.dataset_path = dataset_path
        self.logger = get_logger("DataHandler")
        self._backup_path = None
    
    # ========================================
    # FILE OPERATIONS
    # ========================================
    
    def load_csv(self, create_if_not_exists: bool = True) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Load the dataset from CSV file.
        
        Args:
            create_if_not_exists: Whether to create file if it doesn't exist
        
        Returns:
            Tuple of (success, dataframe, error_message)
        """
        try:
            # Check if file exists
            if not self.dataset_path.exists():
                if create_if_not_exists:
                    self.logger.warning(f"Dataset not found at {self.dataset_path}. Creating new dataset.")
                    return self._create_empty_dataset()
                else:
                    return False, None, f"Dataset not found at {self.dataset_path}"
            
            # Load the CSV file
            df = pd.read_csv(
                self.dataset_path,
                delimiter=CSV_DELIMITER,
                encoding=CSV_ENCODING
            )
            
            # Log the event
            log_dataset_loaded(
                file_path=str(self.dataset_path),
                rows=len(df),
                columns=len(df.columns)
            )
            
            self.logger.info(f"Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")
            
            return True, df, None
            
        except Exception as e:
            error_msg = f"Failed to load dataset: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, None, error_msg
    
    def save_csv(self, df: pd.DataFrame, backup: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Save dataframe to CSV file.
        
        Args:
            df: Dataframe to save
            backup: Whether to create a backup before saving
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Create backup if requested
            if backup and self.dataset_path.exists():
                self._create_backup()
            
            # Ensure directory exists
            self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df.to_csv(
                self.dataset_path,
                index=False,
                sep=CSV_DELIMITER,
                encoding=CSV_ENCODING
            )
            
            self.logger.info(f"Dataset saved successfully: {len(df)} rows")
            return True, None
            
        except Exception as e:
            error_msg = f"Failed to save dataset: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def _create_empty_dataset(self) -> Tuple[bool, pd.DataFrame, Optional[str]]:
        """
        Create an empty dataset with proper structure.
        
        Returns:
            Tuple of (success, dataframe, error_message)
        """
        try:
            # Define all columns
            all_columns = ['ID'] + FEATURE_COLUMNS + [TARGET_COLUMN]
            
            # Create empty dataframe
            df = pd.DataFrame(columns=all_columns)
            
            # Save the empty dataset
            self.save_csv(df, backup=False)
            
            self.logger.info("Created new empty dataset")
            return True, df, None
            
        except Exception as e:
            error_msg = f"Failed to create empty dataset: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, None, error_msg
    
    def _create_backup(self) -> None:
        """Create a backup of the current dataset."""
        try:
            if not self.dataset_path.exists():
                return
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"utility_usage_backup_{timestamp}.csv"
            backup_path = self.dataset_path.parent / backup_filename
            
            # Copy file
            shutil.copy2(self.dataset_path, backup_path)
            
            self.logger.info(f"Backup created: {backup_path}")
            
            # Keep only last 5 backups
            self._cleanup_old_backups(keep_count=5)
            
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {str(e)}")
    
    def _cleanup_old_backups(self, keep_count: int = 5) -> None:
        """
        Remove old backup files, keeping only the most recent ones.
        
        Args:
            keep_count: Number of backup files to keep
        """
        try:
            backup_files = sorted(
                self.dataset_path.parent.glob("utility_usage_backup_*.csv"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Remove old backups
            for backup_file in backup_files[keep_count:]:
                backup_file.unlink()
                self.logger.debug(f"Removed old backup: {backup_file.name}")
                
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old backups: {str(e)}")
    
    # ========================================
    # CRUD OPERATIONS
    # ========================================
    
    def add_record(self, record_data: Dict[str, Any]) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Add a new record to the dataset.
        
        Args:
            record_data: Dictionary containing record data
        
        Returns:
            Tuple of (success, record_id, error_message)
        """
        try:
            # Load current dataset
            success, df, error = self.load_csv(create_if_not_exists=True)
            if not success:
                return False, None, error
            
            # Generate new ID
            new_id = self._generate_id(df)
            record_data['ID'] = new_id
            
            # Ensure all columns are present
            for col in df.columns:
                if col not in record_data:
                    record_data[col] = None
            
            # Add record to dataframe
            new_row = pd.DataFrame([record_data])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Save updated dataset
            success, error = self.save_csv(df)
            if not success:
                return False, None, error
            
            # Log the event
            log_record_added(new_id, str(record_data))
            
            return True, new_id, None
            
        except Exception as e:
            error_msg = f"Failed to add record: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, None, error_msg
    
    def update_record(self, record_id: int, updates: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Update an existing record.
        
        Args:
            record_id: ID of the record to update
            updates: Dictionary containing fields to update
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Load current dataset
            success, df, error = self.load_csv()
            if not success:
                return False, error
            
            # Check if record exists
            if record_id not in df['ID'].values:
                return False, f"Record with ID {record_id} not found"
            
            # Get old values for logging
            old_values = df[df['ID'] == record_id].iloc[0].to_dict()
            
            # Update the record
            for column, value in updates.items():
                if column in df.columns:
                    df.loc[df['ID'] == record_id, column] = value
            
            # Save updated dataset
            success, error = self.save_csv(df)
            if not success:
                return False, error
            
            # Log the event
            changes = {k: f"{old_values.get(k)} -> {v}" for k, v in updates.items()}
            log_record_updated(record_id, str(changes))
            
            return True, None
            
        except Exception as e:
            error_msg = f"Failed to update record: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def delete_record(self, record_id: int) -> Tuple[bool, Optional[str]]:
        """
        Delete a record from the dataset.
        
        Args:
            record_id: ID of the record to delete
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Load current dataset
            success, df, error = self.load_csv()
            if not success:
                return False, error
            
            # Check if record exists
            if record_id not in df['ID'].values:
                return False, f"Record with ID {record_id} not found"
            
            # Delete the record
            df = df[df['ID'] != record_id].reset_index(drop=True)
            
            # Save updated dataset
            success, error = self.save_csv(df)
            if not success:
                return False, error
            
            # Log the event
            log_record_deleted(record_id)
            
            return True, None
            
        except Exception as e:
            error_msg = f"Failed to delete record: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def search_record(self, record_id: int) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Search for a record by ID.
        
        Args:
            record_id: ID of the record to search
        
        Returns:
            Tuple of (success, record_data, error_message)
        """
        try:
            # Load current dataset
            success, df, error = self.load_csv()
            if not success:
                return False, None, error
            
            # Search for the record
            record = df[df['ID'] == record_id]
            
            if record.empty:
                return False, None, f"Record with ID {record_id} not found"
            
            # Convert to dictionary
            record_data = record.iloc[0].to_dict()
            
            return True, record_data, None
            
        except Exception as e:
            error_msg = f"Failed to search record: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, None, error_msg
    
    def get_all_records(self, limit: Optional[int] = None) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Get all records from the dataset.
        
        Args:
            limit: Maximum number of records to return (None for all)
        
        Returns:
            Tuple of (success, dataframe, error_message)
        """
        try:
            # Load current dataset
            success, df, error = self.load_csv()
            if not success:
                return False, None, error
            
            # Apply limit if specified
            if limit and limit > 0:
                df = df.head(limit)
            
            return True, df, None
            
        except Exception as e:
            error_msg = f"Failed to get records: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, None, error_msg
    
    def get_record_count(self) -> Tuple[int, Optional[str]]:
        """
        Get the total number of records in the dataset.
        
        Returns:
            Tuple of (count, error_message)
        """
        try:
            success, df, error = self.load_csv()
            if not success:
                return 0, error
            
            return len(df), None
            
        except Exception as e:
            error_msg = f"Failed to get record count: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return 0, error_msg
    
    def record_exists(self, record_id: int) -> bool:
        """
        Check if a record exists.
        
        Args:
            record_id: ID of the record to check
        
        Returns:
            True if record exists, False otherwise
        """
        try:
            success, df, error = self.load_csv()
            if not success:
                return False
            
            return record_id in df['ID'].values
            
        except Exception:
            return False
    
    def get_existing_ids(self) -> set:
        """
        Get set of all existing record IDs.
        
        Returns:
            Set of record IDs
        """
        try:
            success, df, error = self.load_csv()
            if not success:
                return set()
            
            return set(df['ID'].values) if 'ID' in df.columns else set()
            
        except Exception:
            return set()
    
    # ========================================
    # DISPLAY UTILITIES
    # ========================================
    
    def display_records(self, df: pd.DataFrame, max_records: int = 100) -> None:
        """
        Display records in a formatted table.
        
        Args:
            df: Dataframe to display
            max_records: Maximum number of records to display
        """
        if df is None or df.empty:
            print_info("No records to display")
            return
        
        # Limit records
        display_df = df.head(max_records)
        
        # Prepare headers
        headers = list(display_df.columns)
        
        # Prepare rows
        rows = []
        for _, row in display_df.iterrows():
            row_data = []
            for col in headers:
                value = row[col]
                # Format numeric values
                if isinstance(value, float):
                    value = format_number(value, 2)
                row_data.append(value)
            rows.append(row_data)
        
        # Print table
        print_table(
            headers=headers,
            rows=rows,
            title=f"Records (showing {len(rows)} of {len(df)})",
            max_width=15
        )
        
        if len(df) > max_records:
            print_info(f"Displaying first {max_records} records. Total records: {len(df)}")
    
    def display_record_details(self, record: Dict[str, Any]) -> None:
        """
        Display detailed information for a single record.
        
        Args:
            record: Dictionary containing record data
        """
        if not record:
            print_error("No record data to display")
            return
        
        print("\n" + "=" * 60)
        print("RECORD DETAILS")
        print("=" * 60)
        
        for key, value in record.items():
            if isinstance(value, float):
                value = format_number(value, 4)
            print(f"{key:25s}: {value}")
        
        print("=" * 60)
    
    # ========================================
    # UTILITY METHODS
    # ========================================
    
    def _generate_id(self, df: pd.DataFrame) -> int:
        """
        Generate a new unique ID for a record.
        
        Args:
            df: Current dataframe
        
        Returns:
            New unique ID
        """
        if df.empty or 'ID' not in df.columns:
            return 1
        
        # Get max ID and add 1
        max_id = df['ID'].max()
        return int(max_id) + 1 if pd.notna(max_id) else 1
    
    def export_to_csv(self, output_path: Path, df: Optional[pd.DataFrame] = None) -> Tuple[bool, Optional[str]]:
        """
        Export dataset to a new CSV file.
        
        Args:
            output_path: Path to save the exported CSV
            df: Dataframe to export (None to use current dataset)
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Load dataset if not provided
            if df is None:
                success, df, error = self.load_csv()
                if not success:
                    return False, error
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df.to_csv(
                output_path,
                index=False,
                sep=CSV_DELIMITER,
                encoding=CSV_ENCODING
            )
            
            self.logger.info(f"Dataset exported to: {output_path}")
            return True, None
            
        except Exception as e:
            error_msg = f"Failed to export dataset: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        try:
            success, df, error = self.load_csv()
            if not success:
                return {"error": error}
            
            info = {
                "file_path": str(self.dataset_path),
                "file_exists": self.dataset_path.exists(),
                "total_records": len(df),
                "total_columns": len(df.columns),
                "columns": list(df.columns),
                "memory_usage": f"{df.memory_usage().sum() / 1024**2:.2f} MB",
                "duplicate_rows": df.duplicated().sum(),
                "missing_values": df.isnull().sum().sum()
            }
            
            return info
            
        except Exception as e:
            return {"error": str(e)}
    
    def validate_dataset(self) -> Tuple[bool, List[str]]:
        """
        Validate the dataset structure and data.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Load dataset
            success, df, error = self.load_csv()
            if not success:
                return False, [error]
            
            # Check if empty
            if df.empty:
                issues.append("Dataset is empty")
            
            # Check required columns
            required_columns = ['ID'] + FEATURE_COLUMNS + [TARGET_COLUMN]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                issues.append(f"Missing columns: {', '.join(missing_columns)}")
            
            # Check for duplicate IDs
            if 'ID' in df.columns and df['ID'].duplicated().any():
                issues.append("Duplicate IDs found")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            missing_cols = missing_values[missing_values > 0]
            if not missing_cols.empty:
                issues.append(f"Missing values in columns: {', '.join(missing_cols.index.tolist())}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================


def get_data_handler() -> DataHandler:
    """
    Get a DataHandler instance.
    
    Returns:
        DataHandler instance
    """
    return DataHandler()


def load_dataset() -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
    """
    Load the dataset (convenience function).
    
    Returns:
        Tuple of (success, dataframe, error_message)
    """
    handler = get_data_handler()
    return handler.load_csv()


def save_dataset(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Save the dataset (convenience function).
    
    Args:
        df: Dataframe to save
    
    Returns:
        Tuple of (success, error_message)
    """
    handler = get_data_handler()
    return handler.save_csv(df)