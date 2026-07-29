"""
Data Handler Module
Handles all CSV operations including loading, saving, searching, adding, 
updating, deleting records, and dataset management.
"""

import csv
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .config import DATASET_PATH
from .logger import app_logger, log_record_operation, log_csv_loaded, log_error, log_warning
from .utils import export_to_csv, read_csv_file, print_success, print_error, print_info
from .validation import validate_student_record, ValidationError


class DataHandler:
    """
    Handles all data operations for the Student Performance Prediction System.
    """
    
    def __init__(self, dataset_path: Path = DATASET_PATH):
        """
        Initialize DataHandler.
        
        Args:
            dataset_path: Path to the CSV dataset file
        """
        self.dataset_path = dataset_path
        self.records: List[Dict[str, str]] = []
        self.headers: List[str] = []
        self._load_dataset()
    
    def _load_dataset(self) -> bool:
        """
        Load dataset from CSV file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not self.dataset_path.exists():
                app_logger.warning(f"Dataset file not found: {self.dataset_path}")
                print_error(f"Dataset file not found: {self.dataset_path}")
                print_info("Creating new empty dataset...")
                self.records = []
                self.headers = [
                    'Record_ID', 'Hours_Studied', 'Attendance', 'Parental_Involvement',
                    'Access_to_Resources', 'Extracurricular_Activities', 'Sleep_Hours',
                    'Previous_Scores', 'Motivation_Level', 'Internet_Access',
                    'Tutoring_Sessions', 'Family_Income', 'Teacher_Quality',
                    'School_Type', 'Peer_Influence', 'Physical_Activity',
                    'Learning_Disabilities', 'Parental_Education_Level',
                    'Distance_from_Home', 'Gender', 'Exam_Score'
                ]
                self._save_dataset()
                return True
            
            self.records, self.headers = read_csv_file(self.dataset_path)
            log_csv_loaded(self.dataset_path, len(self.records))
            print_success(f"Loaded {len(self.records)} records from dataset")
            return True
        
        except Exception as e:
            log_error(f"Failed to load dataset: {self.dataset_path}", e)
            print_error(f"Failed to load dataset: {e}")
            return False
    
    def _save_dataset(self) -> bool:
        """
        Save current records to CSV file.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure parent directory exists
            self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.dataset_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.headers)
                writer.writeheader()
                writer.writerows(self.records)
            
            return True
        
        except Exception as e:
            log_error(f"Failed to save dataset: {e}")
            print_error(f"Failed to save dataset: {e}")
            return False
    
    def reload_dataset(self) -> bool:
        """
        Reload dataset from file.
        
        Returns:
            True if reloaded successfully, False otherwise
        """
        return self._load_dataset()
    
    def get_record_count(self) -> int:
        """
        Get total number of records.
        
        Returns:
            Number of records in dataset
        """
        return len(self.records)
    
    def get_headers(self) -> List[str]:
        """
        Get dataset headers.
        
        Returns:
            List of column headers
        """
        return self.headers.copy()
    
    def search_records(
        self,
        search_field: str,
        search_value: str,
        exact_match: bool = False
    ) -> List[Dict[str, str]]:
        """
        Search records by field value.
        
        Args:
            search_field: Field name to search in
            search_value: Value to search for
            exact_match: Whether to use exact match or partial match
        
        Returns:
            List of matching records
        """
        if search_field not in self.headers:
            log_warning(f"Invalid search field: {search_field}")
            return []
        
        results = []
        search_value_lower = search_value.lower().strip()
        
        for record in self.records:
            record_value = str(record.get(search_field, '')).lower()
            
            if exact_match:
                if record_value == search_value_lower:
                    results.append(record)
            else:
                if search_value_lower in record_value:
                    results.append(record)
        
        app_logger.info(f"Search completed: {len(results)} records found for {search_field}={search_value}")
        return results
    
    def search_by_multiple_fields(
        self,
        search_criteria: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """
        Search records by multiple field criteria.
        
        Args:
            search_criteria: Dictionary of field:value pairs
        
        Returns:
            List of matching records
        """
        results = self.records.copy()
        
        for field, value in search_criteria.items():
            if field not in self.headers:
                continue
            
            value_lower = value.lower().strip()
            results = [
                record for record in results
                if value_lower in str(record.get(field, '')).lower()
            ]
        
        app_logger.info(f"Multi-field search: {len(results)} records found")
        return results
    
    def add_record(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Add a new record to the dataset.
        
        Args:
            record: Dictionary containing record data
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Validate record only if it has the required fields for validation
            if all(field in record for field in ['Hours_Studied', 'Attendance', 'Sleep_Hours', 
                                                  'Previous_Scores', 'Tutoring_Sessions', 
                                                  'Physical_Activity', 'Gender', 'School_Type', 
                                                  'Teacher_Quality']):
                is_valid, errors = validate_student_record(record)
                if not is_valid:
                    error_msg = "; ".join(errors)
                    log_warning(f"Record validation failed: {error_msg}")
                    return False, f"Validation failed: {error_msg}"
            
            # Ensure all headers are present
            new_record = {}
            for header in self.headers:
                new_record[header] = str(record.get(header, ''))
            
            # Add timestamp or ID if needed
            if 'Record_ID' not in new_record or not new_record['Record_ID']:
                new_record['Record_ID'] = str(uuid.uuid4())[:8]
            
            # Add to records
            self.records.append(new_record)
            
            # Save to file
            if self._save_dataset():
                log_record_operation("Added", new_record.get('Record_ID', 'N/A'))
                return True, "Record added successfully"
            else:
                # Rollback
                self.records.pop()
                return False, "Failed to save dataset"
        
        except Exception as e:
            log_error("Failed to add record", e)
            return False, f"Error adding record: {str(e)}"
    
    def update_record(
        self,
        record_index: int,
        updated_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Update an existing record.
        
        Args:
            record_index: Index of record to update
            updated_data: Dictionary containing updated field values
        
        Returns:
            Tuple of (success, message)
        """
        try:
            if record_index < 0 or record_index >= len(self.records):
                return False, "Invalid record index"
            
            # Get existing record
            existing_record = self.records[record_index]
            
            # Merge updated data
            updated_record = existing_record.copy()
            for key, value in updated_data.items():
                if key in self.headers:
                    updated_record[key] = str(value)
            
            # Validate updated record
            is_valid, errors = validate_student_record(updated_record)
            if not is_valid:
                error_msg = "; ".join(errors)
                log_warning(f"Record validation failed: {error_msg}")
                return False, f"Validation failed: {error_msg}"
            
            # Update record
            self.records[record_index] = updated_record
            
            # Save to file
            if self._save_dataset():
                record_id = updated_record.get('Record_ID', f'Index {record_index}')
                log_record_operation("Updated", record_id)
                return True, "Record updated successfully"
            else:
                # Rollback
                self.records[record_index] = existing_record
                return False, "Failed to save dataset"
        
        except Exception as e:
            log_error("Failed to update record", e)
            return False, f"Error updating record: {str(e)}"
    
    def delete_record(self, record_index: int) -> Tuple[bool, str]:
        """
        Delete a record by index.
        
        Args:
            record_index: Index of record to delete
        
        Returns:
            Tuple of (success, message)
        """
        try:
            if record_index < 0 or record_index >= len(self.records):
                return False, "Invalid record index"
            
            # Get record ID before deletion
            deleted_record = self.records[record_index]
            record_id = deleted_record.get('Record_ID', f'Index {record_index}')
            
            # Remove record
            self.records.pop(record_index)
            
            # Save to file
            if self._save_dataset():
                log_record_operation("Deleted", record_id)
                return True, "Record deleted successfully"
            else:
                # Rollback - this is tricky, we'd need to store the record
                return False, "Failed to save dataset"
        
        except Exception as e:
            log_error("Failed to delete record", e)
            return False, f"Error deleting record: {str(e)}"
    
    def get_record_by_index(self, record_index: int) -> Optional[Dict[str, str]]:
        """
        Get a record by its index.
        
        Args:
            record_index: Index of record
        
        Returns:
            Record dictionary or None if not found
        """
        if 0 <= record_index < len(self.records):
            return self.records[record_index].copy()
        return None
    
    def get_all_records(self) -> List[Dict[str, str]]:
        """
        Get all records.
        
        Returns:
            List of all records
        """
        return [record.copy() for record in self.records]
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the dataset.
        
        Returns:
            Dictionary containing dataset summary
        """
        summary = {
            'total_records': len(self.records),
            'total_columns': len(self.headers),
            'headers': self.headers,
            'file_path': str(self.dataset_path),
            'file_exists': self.dataset_path.exists(),
            'last_modified': datetime.fromtimestamp(
                self.dataset_path.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M:%S") if self.dataset_path.exists() else "N/A"
        }
        
        # Calculate statistics for numeric columns
        numeric_stats = {}
        numeric_columns = [
            'Hours_Studied', 'Attendance', 'Sleep_Hours',
            'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score'
        ]
        
        for column in numeric_columns:
            if column in self.headers:
                values = []
                for record in self.records:
                    try:
                        values.append(float(record.get(column, 0)))
                    except (ValueError, TypeError):
                        pass
                
                if values:
                    numeric_stats[column] = {
                        'count': len(values),
                        'mean': round(sum(values) / len(values), 2),
                        'min': min(values),
                        'max': max(values)
                    }
        
        summary['numeric_statistics'] = numeric_stats
        
        # Count categorical values
        categorical_stats = {}
        categorical_columns = ['Gender', 'School_Type', 'Teacher_Quality']
        
        for column in categorical_columns:
            if column in self.headers:
                value_counts = {}
                for record in self.records:
                    value = record.get(column, 'Unknown')
                    value_counts[value] = value_counts.get(value, 0) + 1
                categorical_stats[column] = value_counts
        
        summary['categorical_statistics'] = categorical_stats
        
        return summary
    
    def print_dataset_info(self) -> None:
        """
        Print formatted dataset information.
        """
        from .utils import print_header, print_subheader, print_table, print_separator
        
        print_header("DATASET INFORMATION")
        
        summary = self.get_dataset_summary()
        
        print(f"\nFile Path: {summary['file_path']}")
        print(f"File Exists: {'Yes' if summary['file_exists'] else 'No'}")
        print(f"Last Modified: {summary['last_modified']}")
        print(f"Total Records: {summary['total_records']}")
        print(f"Total Columns: {summary['total_columns']}")
        
        print_subheader("COLUMNS")
        for i, header in enumerate(summary['headers'], 1):
            print(f"{i:2d}. {header}")
        
        if summary['numeric_statistics']:
            print_subheader("NUMERIC COLUMNS STATISTICS")
            
            for column, stats in summary['numeric_statistics'].items():
                print(f"\n{column}:")
                print(f"  Count: {stats['count']}")
                print(f"  Mean: {stats['mean']}")
                print(f"  Min: {stats['min']}")
                print(f"  Max: {stats['max']}")
        
        if summary['categorical_statistics']:
            print_subheader("CATEGORICAL COLUMNS DISTRIBUTION")
            
            for column, value_counts in summary['categorical_statistics'].items():
                print(f"\n{column}:")
                for value, count in sorted(value_counts.items()):
                    percentage = (count / summary['total_records']) * 100
                    print(f"  {value}: {count} ({percentage:.1f}%)")
        
        print_separator()
    
    def export_dataset(self, output_path: Path, format: str = 'csv') -> Tuple[bool, str]:
        """
        Export dataset to a file.
        
        Args:
            output_path: Path to save exported file
            format: Export format ('csv' or 'json')
        
        Returns:
            Tuple of (success, message)
        """
        try:
            if format.lower() == 'csv':
                success = export_to_csv(self.records, output_path, self.headers)
                if success:
                    return True, f"Dataset exported to {output_path}"
                else:
                    return False, "Failed to export dataset"
            
            elif format.lower() == 'json':
                import json
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as jsonfile:
                    json.dump({
                        'headers': self.headers,
                        'records': self.records,
                        'exported_at': datetime.now().isoformat()
                    }, jsonfile, indent=2)
                
                return True, f"Dataset exported to {output_path}"
            
            else:
                return False, f"Unsupported format: {format}"
        
        except Exception as e:
            log_error(f"Failed to export dataset: {e}")
            return False, f"Error exporting dataset: {str(e)}"
    
    def get_records_table_data(self, records: Optional[List[Dict[str, str]]] = None) -> Tuple[List[str], List[List[str]]]:
        """
        Get records in table format for display.
        
        Args:
            records: List of records (uses all records if None)
        
        Returns:
            Tuple of (headers, rows)
        """
        if records is None:
            records = self.records
        
        # Select key columns for display
        display_columns = [
            'Record_ID', 'Gender', 'School_Type', 'Hours_Studied',
            'Attendance', 'Previous_Scores', 'Exam_Score'
        ]
        
        # Filter to available columns
        display_columns = [col for col in display_columns if col in self.headers]
        
        # Create headers
        headers = display_columns
        
        # Create rows
        rows = []
        for record in records:
            row = [str(record.get(col, 'N/A')) for col in display_columns]
            rows.append(row)
        
        return headers, rows
    
    def validate_dataset_integrity(self) -> Tuple[bool, List[str]]:
        """
        Validate dataset integrity.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if dataset is empty
        if not self.records:
            issues.append("Dataset is empty")
        
        # Check for missing headers
        if not self.headers:
            issues.append("No headers found in dataset")
        
        # Check for required columns
        required_columns = [
            'Hours_Studied', 'Attendance', 'Sleep_Hours',
            'Previous_Scores', 'Exam_Score'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.headers]
        if missing_columns:
            issues.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check for empty records
        empty_records = sum(1 for record in self.records if not any(record.values()))
        if empty_records > 0:
            issues.append(f"Found {empty_records} empty records")
        
        # Check for duplicate Record_IDs
        if 'Record_ID' in self.headers:
            record_ids = [record.get('Record_ID') for record in self.records]
            duplicate_ids = [id for id in set(record_ids) if record_ids.count(id) > 1]
            if duplicate_ids:
                issues.append(f"Found duplicate Record_IDs: {', '.join(duplicate_ids)}")
        
        return len(issues) == 0, issues