"""
Validation Module Tests

This module contains unit tests for the validation module functions.

Author: CodeVedX AI/ML Internship
"""

import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from validation import (
    validate_integer,
    validate_float,
    validate_menu_choice,
    validate_string,
    validate_negative_values,
    validate_voltage,
    validate_intensity,
    validate_active_power,
    validate_year,
    validate_month,
    validate_day,
    validate_hour,
    is_valid_email,
    is_valid_phone,
    sanitize_string
)


class TestValidateInteger(unittest.TestCase):
    """Test cases for validate_integer function."""
    
    def test_valid_integer(self):
        """Test valid integer input."""
        valid, value, error = validate_integer("25", min_value=0, max_value=100)
        self.assertTrue(valid)
        self.assertEqual(value, 25)
        self.assertIsNone(error)
    
    def test_integer_with_spaces(self):
        """Test integer input with leading/trailing spaces."""
        valid, value, error = validate_integer("  42  ", min_value=0, max_value=100)
        self.assertTrue(valid)
        self.assertEqual(value, 42)
        self.assertIsNone(error)
    
    def test_invalid_integer_string(self):
        """Test invalid integer string."""
        valid, value, error = validate_integer("abc", min_value=0, max_value=100)
        self.assertFalse(valid)
        self.assertIsNone(value)
        self.assertIsNotNone(error)
    
    def test_empty_string(self):
        """Test empty string input."""
        valid, value, error = validate_integer("", min_value=0, max_value=100)
        self.assertFalse(valid)
        self.assertIsNone(value)
        self.assertIn("cannot be empty", error)
    
    def test_below_minimum(self):
        """Test value below minimum."""
        valid, value, error = validate_integer("5", min_value=10, max_value=100)
        self.assertFalse(valid)
        self.assertIsNone(value)
        self.assertIn("must be >= 10", error)
    
    def test_above_maximum(self):
        """Test value above maximum."""
        valid, value, error = validate_integer("150", min_value=0, max_value=100)
        self.assertFalse(valid)
        self.assertIsNone(value)
        self.assertIn("must be <= 100", error)
    
    def test_negative_integer(self):
        """Test negative integer."""
        valid, value, error = validate_integer("-5", min_value=0, max_value=100)
        self.assertFalse(valid)
        self.assertIsNone(value)
        self.assertIn("must be >= 0", error)


class TestValidateFloat(unittest.TestCase):
    """Test cases for validate_float function."""
    
    def test_valid_float(self):
        """Test valid float input."""
        valid, value, error = validate_float("3.14", min_value=0.0, max_value=10.0)
        self.assertTrue(valid)
        self.assertAlmostEqual(value, 3.14)
        self.assertIsNone(error)
    
    def test_integer_as_float(self):
        """Test integer input accepted as float."""
        valid, value, error = validate_float("42", min_value=0.0, max_value=100.0)
        self.assertTrue(valid)
        self.assertAlmostEqual(value, 42.0)
        self.assertIsNone(error)
    
    def test_invalid_float_string(self):
        """Test invalid float string."""
        valid, value, error = validate_float("abc", min_value=0.0, max_value=10.0)
        self.assertFalse(valid)
        self.assertIsNone(value)
        self.assertIsNotNone(error)
    
    def test_negative_float_not_allowed(self):
        """Test negative float when not allowed."""
        valid, value, error = validate_float("-5.5", min_value=0.0, max_value=10.0)
        self.assertFalse(valid)
        self.assertIsNone(value)
        self.assertIn("cannot be negative", error)
    
    def test_negative_float_allowed(self):
        """Test negative float when allowed."""
        valid, value, error = validate_float("-5.5", min_value=-10.0, max_value=10.0, allow_negative=True)
        self.assertTrue(valid)
        self.assertAlmostEqual(value, -5.5)
        self.assertIsNone(error)
    
    def test_below_minimum(self):
        """Test value below minimum."""
        valid, value, error = validate_float("5.0", min_value=10.0, max_value=100.0)
        self.assertFalse(valid)
        self.assertIsNone(value)
        self.assertIn("must be >= 10.0", error)
    
    def test_above_maximum(self):
        """Test value above maximum."""
        valid, value, error = validate_float("150.0", min_value=0.0, max_value=100.0)
        self.assertFalse(valid)
        self.assertIsNone(value)
        self.assertIn("must be <= 100.0", error)


class TestValidateMenuChoice(unittest.TestCase):
    """Test cases for validate_menu_choice function."""
    
    def test_valid_menu_choice(self):
        """Test valid menu choice."""
        valid, choice, error = validate_menu_choice("3", min_choice=1, max_choice=10)
        self.assertTrue(valid)
        self.assertEqual(choice, 3)
        self.assertIsNone(error)
    
    def test_boundary_minimum(self):
        """Test minimum boundary."""
        valid, choice, error = validate_menu_choice("1", min_choice=1, max_choice=10)
        self.assertTrue(valid)
        self.assertEqual(choice, 1)
    
    def test_boundary_maximum(self):
        """Test maximum boundary."""
        valid, choice, error = validate_menu_choice("10", min_choice=1, max_choice=10)
        self.assertTrue(valid)
        self.assertEqual(choice, 10)
    
    def test_below_minimum(self):
        """Test choice below minimum."""
        valid, choice, error = validate_menu_choice("0", min_choice=1, max_choice=10)
        self.assertFalse(valid)
        self.assertIsNone(choice)
    
    def test_above_maximum(self):
        """Test choice above maximum."""
        valid, choice, error = validate_menu_choice("11", min_choice=1, max_choice=10)
        self.assertFalse(valid)
        self.assertIsNone(choice)


class TestValidateString(unittest.TestCase):
    """Test cases for validate_string function."""
    
    def test_valid_string(self):
        """Test valid string."""
        valid, value, error = validate_string("Hello", min_length=2, max_length=50)
        self.assertTrue(valid)
        self.assertEqual(value, "Hello")
        self.assertIsNone(error)
    
    def test_string_with_spaces(self):
        """Test string with leading/trailing spaces."""
        valid, value, error = validate_string("  World  ", min_length=2, max_length=50)
        self.assertTrue(valid)
        self.assertEqual(value, "World")
        self.assertIsNone(error)
    
    def test_empty_string_not_allowed(self):
        """Test empty string when not allowed."""
        valid, value, error = validate_string("", min_length=1, max_length=50)
        self.assertFalse(valid)
        self.assertIsNone(value)
        self.assertIn("cannot be empty", error)
    
    def test_empty_string_allowed(self):
        """Test empty string when allowed."""
        valid, value, error = validate_string("", min_length=0, max_length=50, allow_empty=True)
        self.assertTrue(valid)
        self.assertEqual(value, "")
        self.assertIsNone(error)
    
    def test_too_short(self):
        """Test string too short."""
        valid, value, error = validate_string("A", min_length=2, max_length=50)
        self.assertFalse(valid)
        self.assertIn("must be at least 2 characters", error)
    
    def test_too_long(self):
        """Test string too long."""
        valid, value, error = validate_string("A" * 60, min_length=1, max_length=50)
        self.assertFalse(valid)
        self.assertIn("must be at most 50 characters", error)
    
    def test_pattern_matching(self):
        """Test pattern matching."""
        valid, value, error = validate_string("test@example.com", pattern=r'^[a-z]+@[a-z]+\.[a-z]+$')
        self.assertTrue(valid)
        self.assertEqual(value, "test@example.com")


class TestValidateNegativeValues(unittest.TestCase):
    """Test cases for validate_negative_values function."""
    
    def test_positive_value_allowed(self):
        """Test positive value when negative not allowed."""
        valid, error = validate_negative_values(5.0, field_name="Value", allow_negative=False)
        self.assertTrue(valid)
        self.assertIsNone(error)
    
    def test_negative_value_not_allowed(self):
        """Test negative value when not allowed."""
        valid, error = validate_negative_values(-5.0, field_name="Value", allow_negative=False)
        self.assertFalse(valid)
        self.assertIn("cannot be negative", error)
    
    def test_negative_value_allowed(self):
        """Test negative value when allowed."""
        valid, error = validate_negative_values(-5.0, field_name="Value", allow_negative=True)
        self.assertTrue(valid)
        self.assertIsNone(error)
    
    def test_zero_value(self):
        """Test zero value."""
        valid, error = validate_negative_values(0.0, field_name="Value", allow_negative=False)
        self.assertTrue(valid)
        self.assertIsNone(error)


class TestDomainSpecificValidation(unittest.TestCase):
    """Test cases for domain-specific validation functions."""
    
    def test_valid_voltage(self):
        """Test valid voltage."""
        valid, value, error = validate_voltage("220.5")
        self.assertTrue(valid)
        self.assertAlmostEqual(value, 220.5)
        self.assertIsNone(error)
    
    def test_invalid_voltage_too_low(self):
        """Test voltage below minimum."""
        valid, value, error = validate_voltage("150")
        self.assertFalse(valid)
        self.assertIn("Invalid voltage", error)
    
    def test_invalid_voltage_too_high(self):
        """Test voltage above maximum."""
        valid, value, error = validate_voltage("300")
        self.assertFalse(valid)
        self.assertIn("Invalid voltage", error)
    
    def test_valid_intensity(self):
        """Test valid intensity."""
        valid, value, error = validate_intensity("15.5")
        self.assertTrue(valid)
        self.assertAlmostEqual(value, 15.5)
        self.assertIsNone(error)
    
    def test_valid_active_power(self):
        """Test valid active power."""
        valid, value, error = validate_active_power("3.5")
        self.assertTrue(valid)
        self.assertAlmostEqual(value, 3.5)
        self.assertIsNone(error)
    
    def test_valid_year(self):
        """Test valid year."""
        valid, value, error = validate_year("2023")
        self.assertTrue(valid)
        self.assertEqual(value, 2023)
        self.assertIsNone(error)
    
    def test_invalid_year_too_old(self):
        """Test year too old."""
        valid, value, error = validate_year("1999")
        self.assertFalse(valid)
        self.assertIn("Invalid year", error)
    
    def test_valid_month(self):
        """Test valid month."""
        valid, value, error = validate_month("6")
        self.assertTrue(valid)
        self.assertEqual(value, 6)
        self.assertIsNone(error)
    
    def test_invalid_month(self):
        """Test invalid month."""
        valid, value, error = validate_month("13")
        self.assertFalse(valid)
        self.assertIn("Invalid month", error)
    
    def test_valid_day(self):
        """Test valid day."""
        valid, value, error = validate_day("15", month=6, year=2023)
        self.assertTrue(valid)
        self.assertEqual(value, 15)
        self.assertIsNone(error)
    
    def test_invalid_day_for_month(self):
        """Test invalid day for month."""
        valid, value, error = validate_day("31", month=4, year=2023)  # April has 30 days
        self.assertFalse(valid)
        self.assertIn("Invalid day", error)
    
    def test_valid_hour(self):
        """Test valid hour."""
        valid, value, error = validate_hour("14")
        self.assertTrue(valid)
        self.assertEqual(value, 14)
        self.assertIsNone(error)
    
    def test_invalid_hour(self):
        """Test invalid hour."""
        valid, value, error = validate_hour("25")
        self.assertFalse(valid)
        self.assertIn("Invalid hour", error)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility validation functions."""
    
    def test_valid_email(self):
        """Test valid email."""
        self.assertTrue(is_valid_email("test@example.com"))
        self.assertTrue(is_valid_email("user.name@domain.co.uk"))
    
    def test_invalid_email(self):
        """Test invalid email."""
        self.assertFalse(is_valid_email("invalid.email"))
        self.assertFalse(is_valid_email("@example.com"))
        self.assertFalse(is_valid_email("test@"))
    
    def test_valid_phone(self):
        """Test valid phone number."""
        self.assertTrue(is_valid_phone("1234567890"))
        self.assertTrue(is_valid_phone("+1-234-567-8900"))
    
    def test_invalid_phone(self):
        """Test invalid phone number."""
        self.assertFalse(is_valid_phone("123"))
        self.assertFalse(is_valid_phone("abc"))
    
    def test_sanitize_string(self):
        """Test string sanitization."""
        result = sanitize_string("Hello@World#123", max_length=20)
        self.assertEqual(result, "HelloWorld123")
        
        result = sanitize_string("A" * 30, max_length=10)
        self.assertEqual(len(result), 10)


if __name__ == '__main__':
    unittest.main()