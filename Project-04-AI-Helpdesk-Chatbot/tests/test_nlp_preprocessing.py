"""Test suite for Module 2: NLP Preprocessing (real behaviour, 21 tests)."""

import sys
import json
from pathlib import Path

import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlp_preprocessor import (  # noqa: E402
    NLPPreprocessor, REQUIRED_COLUMNS, OUTPUT_COLUMNS,
    CANONICAL_INTENTS, CANONICAL_ENTITIES,
)
from prepare_nlp_data import NLPPreparationPipeline  # noqa: E402

INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "faq_dataset.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "faq_nlp_ready.csv"
OUTPUT_JSON = PROJECT_ROOT / "data" / "processed" / "faq_nlp_ready.json"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "nlp_preprocessing_report.txt"
CHART_FILES = ["token_length_distribution.png", "entity_frequency.png",
               "question_length_distribution.png"]
EXPECTED_RECORDS = 294
EXPECTED_INTENTS = 22
MIN_EXAMPLES = 12


@pytest.fixture(scope="session")
def preprocessor():
    """Shared preprocessor instance (real NLTK initialised once)."""
    return NLPPreprocessor(PROJECT_ROOT)


@pytest.fixture(scope="session")
def pipeline_outputs():
    """Run the Module 2 pipeline once; return (input_df, output_df)."""
    pipe = NLPPreparationPipeline(PROJECT_ROOT)
    assert pipe.run(), "Module 2 pipeline did not report success"
    assert pipe.input_df is not None and pipe.output_df is not None
    return pipe.input_df, pipe.output_df


# 1. nlp_preprocessor imports
def test_nlp_preprocessor_imports():
    assert NLPPreprocessor is not None
    assert isinstance(REQUIRED_COLUMNS, list)
    assert isinstance(OUTPUT_COLUMNS, list)
    assert set(CANONICAL_INTENTS) >= {"password_reset", "wifi_problems", "payroll"}
    assert set(CANONICAL_ENTITIES) >= {"password", "wifi", "salary", "laptop"}
    pre = NLPPreprocessor(PROJECT_ROOT)
    for method in ("clean_text", "tokenize_text", "remove_stopwords",
                   "lemmatize_tokens", "normalize_intent", "normalize_entity",
                   "extract_entities", "preprocess_question",
                   "preprocess_record", "preprocess_dataset"):
        assert callable(getattr(pre, method)), f"Missing method: {method}"


# 2. clean_text()
def test_clean_text_basic(preprocessor):
    result = preprocessor.clean_text("  How Do I Reset My Company Password??? ")
    assert result == "how do i reset my company password"


# 3. empty input handling
def test_clean_text_empty_input(preprocessor):
    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text(None) == ""
    assert preprocessor.clean_text("   ") == ""
    assert preprocessor.clean_text(0) == "0"  # safe string conversion


# 4. whitespace normalization
def test_clean_text_whitespace_normalization(preprocessor):
    assert preprocessor.clean_text("a   b\t\tc\n") == "a b c"
    assert preprocessor.clean_text("  lead   trail  ") == "lead trail"


# 5. tokenization
def test_tokenization(preprocessor):
    assert preprocessor.tokenize_text("how do i reset my password") == \
        ["how", "do", "i", "reset", "my", "password"]
    assert preprocessor.tokenize_text("") == []


# 6. stopword processing (and clean_question preservation)
def test_stopword_processing(preprocessor):
    assert preprocessor.remove_stopwords(
        ["how", "do", "i", "reset", "my", "password"]) == ["reset", "password"]
    rec = preprocessor.preprocess_question(
        "How do I reset my password?", "password_reset", "oracle", "password")
    assert rec["clean_question"] == "how do i reset my password"
    assert rec["tokens"] == ["how", "do", "i", "reset", "my", "password"]
    assert rec["filtered_tokens"] == ["reset", "password"]

# ===T2===
