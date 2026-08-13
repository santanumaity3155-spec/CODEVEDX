"""Test suite for Module 1: Internal Helpdesk Dataset Preparation.

These tests validate the final processed dataset produced by
`src/prepare_dataset.py`:
    1. Raw dataset exists
    2. Processed dataset exists
    3. Required columns exist
    4. Dataset is not empty
    5. Questions are not empty
    6. Intents are not empty
    7. Answers are not empty
    8. No duplicate questions
    9. At least 10 examples per intent
    10. Intent labels are valid
    11. Entity format is valid
    12. Record count is reasonable
    13. Processed dataset can be reloaded
    14. Report exists
    15. Charts exist

IMPORTANT: `test_intent_distribution_valid` MUST enforce `min_count >= 10`.
"""

import pytest
import pandas as pd
from pathlib import Path

# Canonical vocabulary (kept in sync with prepare_dataset.py).
CANONICAL_INTENTS = {
    "account_access", "attendance", "contact_information", "email_problems",
    "employee_id", "goodbye", "greetings", "help", "holidays", "hr_support",
    "internet_problems", "laptop_problems", "leave_policy", "office_location",
    "password_reset", "payroll", "salary_information", "security",
    "software_installation", "technical_support", "wifi_problems",
    "working_hours",
}

CANONICAL_ENTITIES = {
    "email", "account", "password", "laptop", "wifi", "software",
    "leave", "salary", "payroll", "internet", "working_hours",
    "employee_id", "holiday", "attendance", "hr", "security",
    "contact", "location", "it_support", "greeting", "goodbye", "help",
}

REQUIRED_COLUMNS = ["question", "intent", "answer", "entity"]


class TestDatasetPreparation:
    def setup_method(self):
        self.project_root = Path(__file__).parent.parent
        self.raw_data_path = (
            self.project_root / "data" / "raw" / "faq_dataset.csv"
        )
        self.processed_data_path = (
            self.project_root / "data" / "processed" / "faq_dataset.csv"
        )
        self.report_path = (
            self.project_root / "outputs" / "reports" / "dataset_report.txt"
        )
        self.charts_dir = self.project_root / "outputs" / "charts"

    def _load_processed(self):
        return pd.read_csv(self.processed_data_path)

    # 1. Raw dataset exists
    def test_raw_dataset_exists(self):
        assert self.raw_data_path.exists(), "Raw dataset not found"

    # 2. Processed dataset exists
    def test_processed_dataset_exists(self):
        assert self.processed_data_path.exists(), "Processed dataset not found"

    # 3. Required columns exist
    def test_required_columns_exist(self):
        df = self._load_processed()
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Required column '{col}' missing"

    # 4. Dataset is not empty
    def test_dataset_not_empty(self):
        df = self._load_processed()
        assert len(df) > 0, "Processed dataset is empty"

    # 5. Questions are not empty
    def test_questions_not_empty(self):
        df = self._load_processed()
        empty = (
            df["question"].isna().sum()
            + (df["question"].astype(str).str.strip() == "").sum()
        )
        assert empty == 0, f"Found {empty} empty questions"

    # 6. Intents are not empty
    def test_intents_not_empty(self):
        df = self._load_processed()
        empty = (
            df["intent"].isna().sum()
            + (df["intent"].astype(str).str.strip() == "").sum()
        )
        assert empty == 0, f"Found {empty} empty intents"

    # 7. Answers are not empty
    def test_answers_not_empty(self):
        df = self._load_processed()
        empty = (
            df["answer"].isna().sum()
            + (df["answer"].astype(str).str.strip() == "").sum()
        )
        assert empty == 0, f"Found {empty} empty answers"
    # 8. No duplicate questions
    def test_no_duplicate_questions(self):
        df = self._load_processed()
        dups = df.duplicated(subset=["question"], keep=False).sum()
        assert dups == 0, f"Found {dups} duplicate questions"

    # 9. At least 10 examples per intent (MUST enforce min_count >= 10)
    def test_intent_distribution_valid(self):
        df = self._load_processed()
        min_count = int(df["intent"].value_counts().min())
        assert min_count >= 10, (
            f"Intent distribution invalid: minimum examples per intent is "
            f"{min_count}, but the required minimum is 10"
        )

    # 10. Intent labels are valid
    def test_intent_labels_valid(self):
        df = self._load_processed()
        invalid = set(df["intent"].astype(str).str.strip()) - CANONICAL_INTENTS
        assert not invalid, f"Invalid intent labels: {sorted(invalid)}"

    # 11. Entity format is valid (empty = no entity; otherwise canonical)
    def test_entity_format_valid(self):
        df = self._load_processed()
        invalid = set()
        for val in df["entity"].astype(str).str.strip():
            if val and val not in CANONICAL_ENTITIES:
                invalid.add(val)
        assert not invalid, f"Invalid entity values: {sorted(invalid)}"

    # 12. Record count is reasonable
    def test_record_count_reasonable(self):
        df = self._load_processed()
        # Expected: 159 base + 135 curated = 294 (every intent >= 12 examples).
        assert 160 <= len(df) <= 400, f"Unexpected record count: {len(df)}"

    # 13. Processed dataset can be reloaded (round-trip)
    def test_processed_dataset_reloadable(self):
        df = pd.read_csv(self.processed_data_path)
        assert len(df) > 0
        tmp = self.project_root / "data" / "processed" / "_reload_check.csv"
        df.to_csv(tmp, index=False)
        df2 = pd.read_csv(tmp)
        assert list(df2.columns) == REQUIRED_COLUMNS
        assert len(df2) == len(df)
        tmp.unlink(missing_ok=True)

    # 14. Report exists and is non-empty
    def test_report_exists(self):
        assert self.report_path.exists(), "Dataset report not found"
        assert self.report_path.stat().st_size > 0, "Dataset report is empty"

    # 15. Charts exist
    def test_charts_exist(self):
        for name in ["intent_distribution.png", "intent_distribution_pie.png"]:
            p = self.charts_dir / name
            assert p.exists(), f"Chart missing: {name}"
            assert p.stat().st_size > 0, f"Chart is empty: {name}"