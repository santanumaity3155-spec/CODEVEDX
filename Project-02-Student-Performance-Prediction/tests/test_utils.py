"""
Tests for Utilities Module
"""

import pytest
import csv
import json
from pathlib import Path
from datetime import datetime
from src.utils import (
    format_currency, format_percentage, format_score,
    get_timestamp, get_filename_timestamp, create_directory,
    ensure_directories, export_to_csv, read_csv_file,
    print_separator, print_header, print_subheader,
    print_success, print_error, print_warning, print_info,
    print_table, get_user_confirmation, clear_screen, pause_screen,
    validate_range, truncate_string, calculate_statistics
)


class TestFormattingFunctions:
    """Test formatting utility functions."""
    
    def test_format_currency(self):
        """Test currency formatting."""
        assert format_currency(1234.56) == "$1,234.56"
        assert format_currency(0) == "$0.00"
        assert format_currency(1000000.99) == "$1,000,000.99"
    
    def test_format_percentage(self):
        """Test percentage formatting."""
        assert format_percentage(75.5) == "75.50%"
        assert format_percentage(100) == "100.00%"
        assert format_percentage(0) == "0.00%"
        
        # Test custom decimal places
        assert format_percentage(75.5, decimal_places=1) == "75.5%"
    
    def test_format_score_excellent(self):
        """Test score formatting for excellent grade."""
        result = format_score(95)
        assert "95.00/100" in result
        assert "A+" in result
    
    def test_format_score_good(self):
        """Test score formatting for good grade."""
        result = format_score(85)
        assert "85.00/100" in result
        assert "A" in result
    
    def test_format_score_average(self):
        """Test score formatting for average grade."""
        result = format_score(70)
        assert "70.00/100" in result
        assert "B" in result
    
    def test_format_score_poor(self):
        """Test score formatting for poor grade."""
        result = format_score(45)
        assert "45.00/100" in result
        assert "F" in result
    
    def test_format_score_custom_max(self):
        """Test score formatting with custom max."""
        result = format_score(85, max_score=200)
        assert "85.00/200" in result


class TestDateTimeFunctions:
    """Test date and time utility functions."""
    
    def test_get_timestamp(self):
        """Test timestamp generation."""
        timestamp = get_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0
        
        # Verify format
        try:
            datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pytest.fail("Timestamp format is incorrect")
    
    def test_get_filename_timestamp(self):
        """Test filename-safe timestamp generation."""
        timestamp = get_filename_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0
        
        # Verify format (should not contain special characters)
        try:
            datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        except ValueError:
            pytest.fail("Filename timestamp format is incorrect")
        
        # Ensure no special characters
        assert "_" in timestamp
        assert ":" not in timestamp
        assert " " not in timestamp


class TestDirectoryFunctions:
    """Test directory utility functions."""
    
    def test_create_directory_new(self, tmp_path):
        """Test creating a new directory."""
        new_dir = tmp_path / "new_directory"
        result = create_directory(new_dir)
        
        assert result is True
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    def test_create_directory_existing(self, tmp_path):
        """Test creating an existing directory."""
        existing_dir = tmp_path / "existing_directory"
        existing_dir.mkdir()
        
        result = create_directory(existing_dir)
        
        assert result is True
        assert existing_dir.exists()
    
    def test_create_directory_nested(self, tmp_path):
        """Test creating nested directories."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        result = create_directory(nested_dir)
        
        assert result is True
        assert nested_dir.exists()
        assert nested_dir.is_dir()
    
    def test_ensure_directories(self, tmp_path, monkeypatch):
        """Test ensure_directories function."""
        # This would normally create PREDICTIONS_DIR and OUTPUTS_DIR
        # We'll just verify it runs without error
        ensure_directories()


class TestCSVFunctions:
    """Test CSV utility functions."""
    
    def test_export_to_csv_success(self, tmp_path):
        """Test successful CSV export."""
        data = [
            {'name': 'Alice', 'age': '25'},
            {'name': 'Bob', 'age': '30'}
        ]
        
        output_file = tmp_path / "test.csv"
        result = export_to_csv(data, output_file)
        
        assert result is True
        assert output_file.exists()
        
        # Verify content
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0]['name'] == 'Alice'
            assert rows[1]['age'] == '30'
    
    def test_export_to_csv_with_fieldnames(self, tmp_path):
        """Test CSV export with custom fieldnames."""
        data = [
            {'name': 'Alice', 'age': '25', 'city': 'NYC'},
            {'name': 'Bob', 'age': '30', 'city': 'LA'}
        ]
        
        output_file = tmp_path / "test.csv"
        fieldnames = ['name', 'city']  # Exclude age
        result = export_to_csv(data, output_file, fieldnames)
        
        assert result is True
        
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            assert 'age' not in reader.fieldnames
            assert 'name' in reader.fieldnames
            assert 'city' in reader.fieldnames
    
    def test_export_to_csv_empty_data(self, tmp_path):
        """Test CSV export with empty data."""
        output_file = tmp_path / "test.csv"
        result = export_to_csv([], output_file)
        
        assert result is False
    
    def test_read_csv_file_success(self, tmp_path):
        """Test successful CSV reading."""
        # Create test CSV
        csv_file = tmp_path / "test.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'age', 'city'])
            writer.writerow(['Alice', '25', 'NYC'])
            writer.writerow(['Bob', '30', 'LA'])
        
        records, headers = read_csv_file(csv_file)
        
        assert len(records) == 2
        assert headers == ['name', 'age', 'city']
        assert records[0]['name'] == 'Alice'
        assert records[1]['city'] == 'LA'
    
    def test_read_csv_file_not_found(self, tmp_path):
        """Test reading non-existent CSV file."""
        non_existent = tmp_path / "nonexistent.csv"
        
        with pytest.raises(FileNotFoundError):
            read_csv_file(non_existent)
    
    def test_read_csv_file_empty(self, tmp_path):
        """Test reading empty CSV file."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")
        
        with pytest.raises(ValueError, match="empty or has no headers"):
            read_csv_file(empty_csv)


class TestPrintFunctions:
    """Test print utility functions."""
    
    def test_print_separator(self, capsys):
        """Test separator printing."""
        print_separator(char="*", length=10)
        captured = capsys.readouterr()
        
        assert captured.out.strip() == "*" * 10
    
    def test_print_header(self, capsys):
        """Test header printing."""
        print_header("TEST HEADER", char="=", length=40)
        captured = capsys.readouterr()
        
        lines = captured.out.strip().split('\n')
        assert len(lines) == 3
        assert all(len(line) == 40 for line in lines)
        assert "TEST HEADER" in captured.out
    
    def test_print_subheader(self, capsys):
        """Test subheader printing."""
        print_subheader("TEST SUBHEADER", char="-", length=40)
        captured = capsys.readouterr()
        
        assert "TEST SUBHEADER" in captured.out
    
    def test_print_success(self, capsys):
        """Test success message printing."""
        print_success("Operation completed")
        captured = capsys.readouterr()
        
        assert "✓" in captured.out
        assert "Operation completed" in captured.out
    
    def test_print_error(self, capsys):
        """Test error message printing."""
        print_error("Something went wrong")
        captured = capsys.readouterr()
        
        assert "✗" in captured.out
        assert "Something went wrong" in captured.out
    
    def test_print_warning(self, capsys):
        """Test warning message printing."""
        print_warning("Be careful")
        captured = capsys.readouterr()
        
        assert "⚠" in captured.out
        assert "Be careful" in captured.out
    
    def test_print_info(self, capsys):
        """Test info message printing."""
        print_info("Here's some info")
        captured = capsys.readouterr()
        
        assert "ℹ" in captured.out
        assert "Here's some info" in captured.out
    
    def test_print_table(self, capsys):
        """Test table printing."""
        headers = ["Name", "Age", "City"]
        rows = [
            ["Alice", "25", "NYC"],
            ["Bob", "30", "LA"]
        ]
        
        print_table(headers, rows)
        captured = capsys.readouterr()
        
        assert "Name" in captured.out
        assert "Age" in captured.out
        assert "Alice" in captured.out
        assert "Bob" in captured.out
    
    def test_print_table_empty(self, capsys):
        """Test table printing with empty data."""
        print_table([], [])
        captured = capsys.readouterr()
        
        assert "No data to display" in captured.out


class TestValidationFunctions:
    """Test validation utility functions."""
    
    def test_validate_range_valid(self):
        """Test valid range validation."""
        is_valid, error = validate_range(50, 0, 100, "Test")
        assert is_valid
        assert error == ""
    
    def test_validate_range_below_min(self):
        """Test range validation below minimum."""
        is_valid, error = validate_range(-5, 0, 100, "Test")
        assert not is_valid
        assert "must be between 0 and 100" in error
    
    def test_validate_range_above_max(self):
        """Test range validation above maximum."""
        is_valid, error = validate_range(150, 0, 100, "Test")
        assert not is_valid
        assert "must be between 0 and 100" in error
    
    def test_truncate_string_short(self):
        """Test truncating short string."""
        result = truncate_string("hello", max_length=10)
        assert result == "hello"
    
    def test_truncate_string_long(self):
        """Test truncating long string."""
        result = truncate_string("hello world", max_length=8)
        assert result == "hello..."
        assert len(result) == 8
    
    def test_calculate_statistics(self):
        """Test statistics calculation."""
        values = [10, 20, 30, 40, 50]
        stats = calculate_statistics(values)
        
        assert stats['mean'] == 30.0
        assert stats['median'] == 30.0
        assert stats['min'] == 10
        assert stats['max'] == 50
        assert 'std' in stats
    
    def test_calculate_statistics_empty(self):
        """Test statistics calculation with empty list."""
        stats = calculate_statistics([])
        assert stats == {}


class TestUserInputFunctions:
    """Test user input utility functions."""
    
    def test_get_user_confirmation_yes(self, monkeypatch, capsys):
        """Test user confirmation with yes."""
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        
        result = get_user_confirmation("Continue?")
        assert result is True
    
    def test_get_user_confirmation_no(self, monkeypatch, capsys):
        """Test user confirmation with no."""
        monkeypatch.setattr('builtins.input', lambda _: 'n')
        
        result = get_user_confirmation("Continue?")
        assert result is False
    
    def test_get_user_confirmation_invalid_then_yes(self, monkeypatch, capsys):
        """Test user confirmation with invalid input then yes."""
        inputs = iter(['invalid', 'yes'])
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))
        
        result = get_user_confirmation("Continue?")
        assert result is True
    
    def test_clear_screen(self, monkeypatch):
        """Test screen clearing."""
        called = []
        
        def mock_system(cmd):
            called.append(cmd)
            return ""
        
        monkeypatch.setattr('os.system', mock_system)
        clear_screen()
        
        assert len(called) == 1
        assert 'cls' in called[0] or 'clear' in called[0]
    
    def test_pause_screen(self, monkeypatch):
        """Test screen pausing."""
        monkeypatch.setattr('builtins.input', lambda _: '')
        
        # Should not raise an exception
        pause_screen("Press Enter...")