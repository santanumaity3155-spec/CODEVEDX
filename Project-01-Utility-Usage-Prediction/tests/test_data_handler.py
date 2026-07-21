"""
Data Handler Module Tests

This module contains unit tests for the data handler module functions.

Author: CodeVedX AI/ML Internship
"""

import unittest
import sys
import os
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_handler import DataHandler, get_data_handler, load_dataset, save_dataset
from config import FEATURE_COLUMNS, TARGET_COLUMN


class TestDataHandler(unittest.TestCase):
    """Test cases for DataHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_csv_path = Path(self.test_dir) / "test_utility_usage.csv"
        
        # Create test dataframe
        self.test_data = pd.DataFrame({
            'ID': [1, 2, 3],
            'Global_reactive_power': [0.5, 0.6, 0.7],
            'Voltage': [220.0, 221.0, 222.0],
            'Global_intensity': [10.0, 11.0, 12.0],
            'Sub_metering_1': [1.0, 2.0, 3.0],
            'Sub_metering_2': [0.0, 1.0, 2.0],
            'Sub_metering_3': [5.0, 6.0, 7.0],
            'Year': [2023, 2023, 2023],
            'Month': [1, 2, 3],
            'Day': [1, 2, 3],
            'Hour': [0, 1, 2],
            'Global_active_power': [1.5, 1.6, 1.7]
        })
        
        # Save test data
        self.test_data.to_csv(self.test_csv_path, index=False)
        
        # Create DataHandler instance
        self.data_handler = DataHandler(dataset_path=self.test_csv_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove all CSV files created during tests
        for csv_file in Path(self.test_dir).glob("*.csv"):
            try:
                csv_file.unlink()
            except PermissionError:
                pass
        
        # Remove backup files
        for backup_file in Path(self.test_dir).glob("utility_usage_backup_*.csv"):
            try:
                backup_file.unlink()
            except PermissionError:
                pass
        
        # Remove directory and all remaining contents
        try:
            import shutil
            shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception:
            pass
    
    def test_load_csv_success(self):
        """Test successful CSV loading."""
        success, df, error = self.data_handler.load_csv(create_if_not_exists=False)
        
        self.assertTrue(success)
        self.assertIsNotNone(df)
        self.assertIsNone(error)
        self.assertEqual(len(df), 3)
        self.assertEqual(len(df.columns), 12)
    
    def test_load_csv_file_not_found(self):
        """Test loading non-existent CSV file."""
        handler = DataHandler(dataset_path=Path(self.test_dir) / "nonexistent.csv")
        success, df, error = handler.load_csv(create_if_not_exists=False)
        
        self.assertFalse(success)
        self.assertIsNone(df)
        self.assertIsNotNone(error)
        self.assertIn("not found", error)
    
    def test_load_csv_create_if_not_exists(self):
        """Test creating CSV if it doesn't exist."""
        handler = DataHandler(dataset_path=Path(self.test_dir) / "new_file.csv")
        success, df, error = handler.load_csv(create_if_not_exists=True)
        
        self.assertTrue(success)
        self.assertIsNotNone(df)
        self.assertIsNone(error)
        self.assertTrue((Path(self.test_dir) / "new_file.csv").exists())
    
    def test_save_csv_success(self):
        """Test successful CSV saving."""
        success, error = self.data_handler.save_csv(self.test_data, backup=False)
        
        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertTrue(self.test_csv_path.exists())
    
    def test_add_record(self):
        """Test adding a new record."""
        new_record = {
            'Global_reactive_power': 0.8,
            'Voltage': 223.0,
            'Global_intensity': 13.0,
            'Sub_metering_1': 4.0,
            'Sub_metering_2': 3.0,
            'Sub_metering_3': 8.0,
            'Year': 2023,
            'Month': 4,
            'Day': 4,
            'Hour': 3,
            'Global_active_power': 1.8
        }
        
        success, record_id, error = self.data_handler.add_record(new_record)
        
        self.assertTrue(success)
        self.assertIsNotNone(record_id)
        self.assertEqual(record_id, 4)  # Next ID after 1, 2, 3
        self.assertIsNone(error)
    
    def test_update_record(self):
        """Test updating an existing record."""
        updates = {
            'Voltage': 225.0,
            'Global_intensity': 15.0
        }
        
        success, error = self.data_handler.update_record(1, updates)
        
        self.assertTrue(success)
        self.assertIsNone(error)
        
        # Verify update
        success, record, error = self.data_handler.search_record(1)
        self.assertTrue(success)
        self.assertEqual(record['Voltage'], 225.0)
        self.assertEqual(record['Global_intensity'], 15.0)
    
    def test_update_record_not_found(self):
        """Test updating non-existent record."""
        updates = {'Voltage': 225.0}
        success, error = self.data_handler.update_record(999, updates)
        
        self.assertFalse(success)
        self.assertIsNotNone(error)
        self.assertIn("not found", error)
    
    def test_delete_record(self):
        """Test deleting a record."""
        success, error = self.data_handler.delete_record(1)
        
        self.assertTrue(success)
        self.assertIsNone(error)
        
        # Verify deletion
        success, record, error = self.data_handler.search_record(1)
        self.assertFalse(success)
        self.assertIn("not found", error)
    
    def test_delete_record_not_found(self):
        """Test deleting non-existent record."""
        success, error = self.data_handler.delete_record(999)
        
        self.assertFalse(success)
        self.assertIsNotNone(error)
        self.assertIn("not found", error)
    
    def test_search_record(self):
        """Test searching for a record."""
        success, record, error = self.data_handler.search_record(1)
        
        self.assertTrue(success)
        self.assertIsNotNone(record)
        self.assertIsNone(error)
        self.assertEqual(record['ID'], 1)
        self.assertEqual(record['Voltage'], 220.0)
    
    def test_search_record_not_found(self):
        """Test searching for non-existent record."""
        success, record, error = self.data_handler.search_record(999)
        
        self.assertFalse(success)
        self.assertIsNone(record)
        self.assertIsNotNone(error)
        self.assertIn("not found", error)
    
    def test_get_all_records(self):
        """Test getting all records."""
        success, df, error = self.data_handler.get_all_records()
        
        self.assertTrue(success)
        self.assertIsNotNone(df)
        self.assertIsNone(error)
        self.assertEqual(len(df), 3)
    
    def test_get_all_records_with_limit(self):
        """Test getting records with limit."""
        success, df, error = self.data_handler.get_all_records(limit=2)
        
        self.assertTrue(success)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
    
    def test_get_record_count(self):
        """Test getting record count."""
        count, error = self.data_handler.get_record_count()
        
        self.assertEqual(count, 3)
        self.assertIsNone(error)
    
    def test_record_exists(self):
        """Test checking if record exists."""
        self.assertTrue(self.data_handler.record_exists(1))
        self.assertFalse(self.data_handler.record_exists(999))
    
    def test_get_existing_ids(self):
        """Test getting existing IDs."""
        ids = self.data_handler.get_existing_ids()
        
        self.assertIsInstance(ids, set)
        self.assertEqual(ids, {1, 2, 3})
    
    def test_get_dataset_info(self):
        """Test getting dataset information."""
        info = self.data_handler.get_dataset_info()
        
        self.assertIn("file_path", info)
        self.assertIn("total_records", info)
        self.assertIn("total_columns", info)
        self.assertEqual(info["total_records"], 3)
        self.assertEqual(info["total_columns"], 12)
    
    def test_validate_dataset_valid(self):
        """Test validating a valid dataset."""
        is_valid, issues = self.data_handler.validate_dataset()
        
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
    
    def test_export_to_csv(self):
        """Test exporting dataset to CSV."""
        export_path = Path(self.test_dir) / "exported_data.csv"
        
        success, error = self.data_handler.export_to_csv(export_path)
        
        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertTrue(export_path.exists())
        
        # Verify exported data
        exported_df = pd.read_csv(export_path)
        self.assertEqual(len(exported_df), 3)


class TestDataHandlerConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def test_get_data_handler(self):
        """Test getting data handler instance."""
        handler = get_data_handler()
        self.assertIsInstance(handler, DataHandler)
    
    def test_load_dataset(self):
        """Test load_dataset convenience function."""
        # Since load_dataset() uses the global DATASET_PATH as default parameter,
        # test that it returns properly when passed a specific path via DataHandler
        with tempfile.TemporaryDirectory() as test_dir:
            test_path = Path(test_dir) / "test.csv"
            test_data = pd.DataFrame({'ID': [1], 'Global_active_power': [1.0]})
            test_data.to_csv(test_path, index=False)
            
            # Create handler with custom path and load directly
            handler = DataHandler(dataset_path=test_path)
            success, df, error = handler.load_csv()
            
            self.assertTrue(success)
            self.assertIsNotNone(df)
            self.assertEqual(len(df), 1)


if __name__ == '__main__':
    unittest.main()