"""
Tests for Data Handler Module
"""

import pytest
import csv
import json
from pathlib import Path
from datetime import datetime
from src.data_handler import DataHandler
from src.config import DATASET_PATH


class TestDataHandlerInitialization:
    """Test DataHandler initialization."""
    
    def test_init_with_default_path(self):
        """Test initialization with default dataset path."""
        handler = DataHandler()
        assert handler.dataset_path == DATASET_PATH
        assert isinstance(handler.records, list)
        assert isinstance(handler.headers, list)
    
    def test_init_with_custom_path(self, tmp_path):
        """Test initialization with custom dataset path."""
        custom_path = tmp_path / "custom_data.csv"
        handler = DataHandler(dataset_path=custom_path)
        assert handler.dataset_path == custom_path
    
    def test_init_creates_new_dataset_if_not_exists(self, tmp_path):
        """Test that initialization creates new dataset if file doesn't exist."""
        new_path = tmp_path / "new_dataset.csv"
        assert not new_path.exists()
        
        handler = DataHandler(dataset_path=new_path)
        
        assert new_path.exists()
        assert len(handler.headers) > 0
        assert len(handler.records) == 0


class TestDataHandlerLoadSave:
    """Test DataHandler load and save operations."""
    
    def test_load_existing_dataset(self, tmp_path):
        """Test loading an existing dataset."""
        # Create a test CSV file
        csv_path = tmp_path / "test.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Age', 'City'])
            writer.writerow(['Alice', '25', 'NYC'])
            writer.writerow(['Bob', '30', 'LA'])
        
        handler = DataHandler(dataset_path=csv_path)
        
        assert len(handler.records) == 2
        assert handler.headers == ['Name', 'Age', 'City']
        assert handler.records[0]['Name'] == 'Alice'
    
    def test_save_dataset(self, tmp_path):
        """Test saving dataset."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        # Add a record
        handler.records = [
            {'Name': 'Alice', 'Age': '25', 'City': 'NYC'},
            {'Name': 'Bob', 'Age': '30', 'City': 'LA'}
        ]
        handler.headers = ['Name', 'Age', 'City']
        
        success = handler._save_dataset()
        
        assert success is True
        assert csv_path.exists()
        
        # Verify content
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
    
    def test_reload_dataset(self, tmp_path):
        """Test reloading dataset."""
        csv_path = tmp_path / "test.csv"
        
        # Create initial dataset
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Age'])
            writer.writerow(['Alice', '25'])
        
        handler = DataHandler(dataset_path=csv_path)
        assert len(handler.records) == 1
        
        # Modify external file
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Age'])
            writer.writerow(['Bob', '30'])
            writer.writerow(['Charlie', '35'])
        
        # Reload
        success = handler.reload_dataset()
        
        assert success is True
        assert len(handler.records) == 2


class TestDataHandlerRecordOperations:
    """Test DataHandler record operations."""
    
    def test_get_record_count(self, tmp_path):
        """Test getting record count."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        assert handler.get_record_count() == 0
        
        handler.records = [
            {'Name': 'Alice'},
            {'Name': 'Bob'},
            {'Name': 'Charlie'}
        ]
        
        assert handler.get_record_count() == 3
    
    def test_get_headers(self, tmp_path):
        """Test getting headers."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        headers = handler.get_headers()
        assert isinstance(headers, list)
        assert headers == handler.headers
    
    def test_get_record_by_index(self, tmp_path):
        """Test getting record by index."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        handler.records = [
            {'Name': 'Alice', 'Age': '25'},
            {'Name': 'Bob', 'Age': '30'},
            {'Name': 'Charlie', 'Age': '35'}
        ]
        
        # Valid index
        record = handler.get_record_by_index(1)
        assert record is not None
        assert record['Name'] == 'Bob'
        
        # Invalid index
        record = handler.get_record_by_index(10)
        assert record is None
        
        record = handler.get_record_by_index(-1)
        assert record is None
    
    def test_get_all_records(self, tmp_path):
        """Test getting all records."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        handler.records = [
            {'Name': 'Alice'},
            {'Name': 'Bob'}
        ]
        
        all_records = handler.get_all_records()
        
        assert len(all_records) == 2
        assert all_records[0]['Name'] == 'Alice'
        assert all_records[1]['Name'] == 'Bob'
        
        # Verify it's a copy, not the original
        all_records[0]['Name'] = 'Modified'
        assert handler.records[0]['Name'] == 'Alice'


class TestDataHandlerSearch:
    """Test DataHandler search operations."""
    
    def test_search_records_exact_match(self, tmp_path):
        """Test exact match search."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        handler.records = [
            {'Name': 'Alice', 'City': 'NYC'},
            {'Name': 'Bob', 'City': 'LA'},
            {'Name': 'Alice', 'City': 'SF'}
        ]
        handler.headers = ['Name', 'City']
        
        results = handler.search_records('Name', 'Alice', exact_match=True)
        
        assert len(results) == 2
    
    def test_search_records_partial_match(self, tmp_path):
        """Test partial match search."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        handler.records = [
            {'Name': 'Alice', 'City': 'NYC'},
            {'Name': 'Bob', 'City': 'LA'},
            {'Name': 'Charlie', 'City': 'SF'}
        ]
        handler.headers = ['Name', 'City']
        
        results = handler.search_records('Name', 'li', exact_match=False)
        
        assert len(results) == 2  # Alice and Charlie
        assert any(r['Name'] == 'Alice' for r in results)
        assert any(r['Name'] == 'Charlie' for r in results)
    
    def test_search_records_invalid_field(self, tmp_path):
        """Test search with invalid field."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        handler.records = [{'Name': 'Alice'}]
        handler.headers = ['Name']
        
        results = handler.search_records('InvalidField', 'Alice')
        
        assert len(results) == 0
    
    def test_search_by_multiple_fields(self, tmp_path):
        """Test multi-field search."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        handler.records = [
            {'Name': 'Alice', 'City': 'NYC', 'Age': '25'},
            {'Name': 'Bob', 'City': 'LA', 'Age': '30'},
            {'Name': 'Charlie', 'City': 'NYC', 'Age': '35'}
        ]
        handler.headers = ['Name', 'City', 'Age']
        
        results = handler.search_by_multiple_fields({
            'City': 'NYC',
            'Age': '25'
        })
        
        assert len(results) == 1
        assert results[0]['Name'] == 'Alice'


class TestDataHandlerAddUpdateDelete:
    """Test DataHandler add, update, and delete operations."""
    
    def test_add_record(self, tmp_path):
        """Test adding a record."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        record = {
            'Name': 'Alice',
            'Age': '25',
            'City': 'NYC'
        }
        
        success, message = handler.add_record(record)
        
        assert success is True
        assert "successfully" in message.lower()
        assert len(handler.records) == 1
        assert handler.records[0]['Name'] == 'Alice'
    
    def test_add_record_validation_failure(self, tmp_path):
        """Test adding a record that fails validation."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        # Invalid record (missing required fields)
        record = {
            'Name': 'Alice'
            # Missing Age and City
        }
        
        success, message = handler.add_record(record)
        
        assert success is False
        assert "validation" in message.lower()
    
    def test_update_record(self, tmp_path):
        """Test updating a record."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        handler.records = [
            {'Name': 'Alice', 'Age': '25', 'City': 'NYC'},
            {'Name': 'Bob', 'Age': '30', 'City': 'LA'}
        ]
        handler.headers = ['Name', 'Age', 'City']
        
        # Update first record
        updated_data = {'Age': '26'}
        success, message = handler.update_record(0, updated_data)
        
        assert success is True
        assert handler.records[0]['Age'] == '26'
        assert handler.records[0]['Name'] == 'Alice'  # Unchanged
    
    def test_update_record_invalid_index(self, tmp_path):
        """Test updating with invalid index."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        handler.records = [{'Name': 'Alice'}]
        
        success, message = handler.update_record(10, {'Name': 'Bob'})
        
        assert success is False
        assert "invalid" in message.lower()
    
    def test_delete_record(self, tmp_path):
        """Test deleting a record."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        handler.records = [
            {'Name': 'Alice', 'Age': '25'},
            {'Name': 'Bob', 'Age': '30'},
            {'Name': 'Charlie', 'Age': '35'}
        ]
        handler.headers = ['Name', 'Age']
        
        success, message = handler.delete_record(1)
        
        assert success is True
        assert len(handler.records) == 2
        assert handler.records[1]['Name'] == 'Charlie'
    
    def test_delete_record_invalid_index(self, tmp_path):
        """Test deleting with invalid index."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        handler.records = [{'Name': 'Alice'}]
        
        success, message = handler.delete_record(10)
        
        assert success is False
        assert "invalid" in message.lower()


class TestDataHandlerSummary:
    """Test DataHandler summary operations."""
    
    def test_get_dataset_summary(self, tmp_path):
        """Test getting dataset summary."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        handler.records = [
            {'Name': 'Alice', 'Age': '25', 'Score': '85'},
            {'Name': 'Bob', 'Age': '30', 'Score': '90'}
        ]
        handler.headers = ['Name', 'Age', 'Score']
        
        summary = handler.get_dataset_summary()
        
        assert summary['total_records'] == 2
        assert summary['total_columns'] == 3
        assert 'headers' in summary
        assert 'numeric_statistics' in summary
    
    def test_validate_dataset_integrity_valid(self, tmp_path):
        """Test dataset integrity validation with valid data."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        handler.records = [{'Name': 'Alice', 'Age': '25'}]
        handler.headers = ['Name', 'Age']
        
        is_valid, issues = handler.validate_dataset_integrity()
        
        assert is_valid is True
        assert len(issues) == 0
    
    def test_validate_dataset_integrity_empty(self, tmp_path):
        """Test dataset integrity validation with empty dataset."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        is_valid, issues = handler.validate_dataset_integrity()
        
        assert is_valid is False
        assert any("empty" in issue.lower() for issue in issues)


class TestDataHandlerExport:
    """Test DataHandler export operations."""
    
    def test_export_dataset_csv(self, tmp_path):
        """Test exporting dataset to CSV."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        handler.records = [
            {'Name': 'Alice', 'Age': '25'},
            {'Name': 'Bob', 'Age': '30'}
        ]
        handler.headers = ['Name', 'Age']
        
        export_path = tmp_path / "export.csv"
        success, message = handler.export_dataset(export_path, format='csv')
        
        assert success is True
        assert export_path.exists()
    
    def test_export_dataset_json(self, tmp_path):
        """Test exporting dataset to JSON."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        handler.records = [
            {'Name': 'Alice', 'Age': '25'},
            {'Name': 'Bob', 'Age': '30'}
        ]
        handler.headers = ['Name', 'Age']
        
        export_path = tmp_path / "export.json"
        success, message = handler.export_dataset(export_path, format='json')
        
        assert success is True
        assert export_path.exists()
        
        # Verify JSON content
        with open(export_path, 'r') as f:
            data = json.load(f)
            assert 'headers' in data
            assert 'records' in data
            assert len(data['records']) == 2
    
    def test_export_dataset_unsupported_format(self, tmp_path):
        """Test exporting with unsupported format."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        export_path = tmp_path / "export.xml"
        success, message = handler.export_dataset(export_path, format='xml')
        
        assert success is False
        assert "unsupported" in message.lower()


class TestDataHandlerTableDisplay:
    """Test DataHandler table display operations."""
    
    def test_get_records_table_data(self, tmp_path):
        """Test getting records in table format."""
        csv_path = tmp_path / "test.csv"
        handler = DataHandler(dataset_path=csv_path)
        
        handler.headers = ['Record_ID', 'Name', 'Age', 'City', 'Score']
        handler.records = [
            {'Record_ID': '1', 'Name': 'Alice', 'Age': '25', 'City': 'NYC', 'Score': '85'},
            {'Record_ID': '2', 'Name': 'Bob', 'Age': '30', 'City': 'LA', 'Score': '90'}
        ]
        
        headers, rows = handler.get_records_table_data()
        
        assert len(headers) > 0
        assert len(rows) == 2
        assert 'Record_ID' in headers
        assert 'Name' in headers