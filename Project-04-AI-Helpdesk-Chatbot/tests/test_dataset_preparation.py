"""Test suite for Module 1: Dataset Preparation"""
import pytest
import pandas as pd
from pathlib import Path


class TestDatasetPreparation:
    """Test dataset preparation and validation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.raw_data_path = self.project_root / "data" / "raw" / "faq_dataset.csv"
        self.processed_data_path = self.project_root / "data" / "processed" / "faq_dataset.csv"
    
    def test_raw_dataset_exists(self):
        """Test 1: Raw dataset exists."""
        assert self.raw_data_path.exists(), "Raw dataset not found"
    
    def test_dataset_loads(self):
        """Test 2: Dataset loads successfully."""
        df = pd.read_csv(self.raw_data_path)
        assert len(df) > 0, "Dataset is empty"
    
    def test_required_columns_exist(self):
        """Test 3: Required columns exist."""
        df = pd.read_csv(self.raw_data_path)
        required_columns = ["question", "intent", "answer"]
        for col in required_columns:
            assert col in df.columns, f"Column {col} missing"
    
    def test_processed_dataset_generated(self):
        """Test 4: Processed dataset is generated."""
        assert self.processed_data_path.exists(), "Processed dataset not found"
    
    def test_processed_dataset_not_empty(self):
        """Test 5: Processed dataset is not empty."""
        df = pd.read_csv(self.processed_data_path)
        assert len(df) > 0, "Processed dataset is empty"
    
    def test_no_missing_question_values(self):
        """Test 6: No missing question values."""
        df = pd.read_csv(self.processed_data_path)
        missing = df["question"].isna().sum()
        assert missing == 0, f"Found {missing} missing questions"
    
    def test_no_missing_intent_values(self):
        """Test 7: No missing intent values."""
        df = pd.read_csv(self.processed_data_path)
        missing = df["intent"].isna().sum()
        assert missing == 0, f"Found {missing} missing intents"
    
    def test_no_missing_answer_values(self):
        """Test 8: No missing answer values."""
        df = pd.read_csv(self.processed_data_path)
        missing = df["answer"].isna().sum()
        assert missing == 0, f"Found {missing} missing answers"
    
    def test_no_duplicate_questions(self):
        """Test 9: No duplicate questions remain."""
        df = pd.read_csv(self.processed_data_path)
        duplicates = df.duplicated(subset=["question"], keep=False).sum()
        assert duplicates == 0, f"Found {duplicates} duplicate questions"
    
    def test_intent_distribution_valid(self):
        """Test 10: Intent distribution is valid."""
        df = pd.read_csv(self.processed_data_path)
        unique_intents = df["intent"].nunique()
        assert unique_intents >= 5, f"Too few intents: {unique_intents}"
        intent_counts = df["intent"].value_counts()
        min_count = intent_counts.min()
        assert min_count >= 1, f"Some intents have too few examples"
