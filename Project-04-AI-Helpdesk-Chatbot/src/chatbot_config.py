"""
Module 4: Chatbot Configuration
==============================

Central, typed configuration for the AI Helpdesk Chatbot (Module 4).  All Module 4
source modules import their tunable parameters from :class:`ChatbotConfig` instead
of scattering hard-coded thresholds and messages across the codebase.

Design notes
------------
* No absolute Windows paths are hard-coded; every path is derived from the
  project root (the parent of ``src/``) so the code works from any checkout.
* The confidence thresholds are calibrated against the deterministic Linear SVM
  produced by Module 3 (``confidence_source == "decision_function_margin"``).
  They are deliberately *configurable* (not magic numbers buried in logic).
* Logging uses a module-level ``NullHandler`` so importing this module is silent
  by default; the CLI / application configures handlers as needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Project paths (derived, never hard-coded absolute paths)
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_FAQ_NLP_READY_CSV = PROJECT_ROOT / "data" / "processed" / "faq_nlp_ready.csv"
DEFAULT_FAQ_DATASET_CSV = PROJECT_ROOT / "data" / "processed" / "faq_dataset.csv"

# Serialized artifacts that *already exist* (produced by Module 3).
DEFAULT_PIPELINE_PATH = DEFAULT_MODEL_DIR / "intent_classifier_pipeline.pkl"
DEFAULT_CLASSIFIER_PATH = DEFAULT_MODEL_DIR / "intent_classifier.pkl"
DEFAULT_VECTORIZER_PATH = DEFAULT_MODEL_DIR / "tfidf_vectorizer.pkl"

# ---------------------------------------------------------------------------
# Confidence thresholds (calibrated against the real Module 3 Linear SVM)
# ---------------------------------------------------------------------------
# Linear SVMs do not expose probabilities, so Module 3 derives a bounded decision-
# margin score via ``sigmoid(margin)``.  Across the helpdesk dataset these scores
# land roughly:
#   * garbage / out-of-domain input : ~0.500-0.510  (no decision margin)
#   * uncertain but helpdesk input  : 0.52 - 0.65
#   * confident helpdesk input      : 0.65 - 0.95+
# ``LOW_CONFIDENCE_THRESHOLD`` therefore separates near-random input from real
# helpdesk questions.  ``MEDIUM_CONFIDENCE_THRESHOLD`` separates uncertain from
# confident answers.
LOW_CONFIDENCE_THRESHOLD: float = 0.52
MEDIUM_CONFIDENCE_THRESHOLD: float = 0.65

# Cosine-relevance of the cleaned query against the fitted FAQ corpus.  Pure
# nonsense or out-of-domain text shares no (or almost no) vocabulary with the
# helpdesk FAQ and scores near 0.0; genuine helpdesk questions score >= ~0.50.
# This is a *secondary* signal computed with the already-fitted TF-IDF vectorizer
# (no refitting -> no data leakage).
OUT_OF_DOMAIN_RELEVANCE_THRESHOLD: float = 0.40

# Input validation limits.
MAX_QUERY_LENGTH: int = 500

# ---------------------------------------------------------------------------
# Fallback messages (helpdesk-oriented, professional)
# ---------------------------------------------------------------------------
FALLBACK_EMPTY_MESSAGE: str = "Please enter a helpdesk question."
FALLBACK_LOW_CONFIDENCE_MESSAGE: str = (
    "I'm not confident I understood your request. Could you rephrase your question?"
)
FALLBACK_OUT_OF_DOMAIN_MESSAGE: str = (
    "I'm not confident I understood your request. Could you rephrase your question?"
)
FALLBACK_NO_ANSWER_MESSAGE: str = (
    "I don't have an answer for that. For immediate assistance, contact IT support "
    "at extension 1234."
)
FALLBACK_OVERFLOW_MESSAGE: str = (
    "That question is too long. Please keep it under 500 characters."
)

# Prefix added to medium-confidence answers to convey appropriate caution.
MEDIUM_CONFIDENCE_NOTE: str = (
    "[Note: This answer was generated with moderate confidence. "
    "If this isn't quite right, please rephrase your question.]"
)

CLI_BANNER: str = (
    "==================================================\n"
    "        AI HELPDESK CHATBOT (Module 4)\n"
    "==================================================\n"
    "Ask a helpdesk question about IT, HR, or company policies.\n"
    "Commands: 'help' (this message), 'clear' (clear screen), "
    "'exit'/'quit' (leave).\n"
)

CLI_HELP_TEXT: str = (
    "Available commands:\n"
    "  <any question>  Ask the helpdesk bot a question.\n"
    "  help            Show this help message.\n"
    "  clear           Clear the conversation screen.\n"
    "  exit / quit     Exit the chatbot.\n"
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("chatbot")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

def get_logger() -> logging.Logger:
    """Return the shared Module 4 logger (silenced by default)."""
    return logger

@dataclass
class ChatbotConfig:
    """Typed, configurable settings for the helpdesk chatbot.

    Attributes
    ----------
    project_root : Path
        Root of the CodeVedX helpdesk project.
    model_dir : Path
        Directory holding the serialized Module 3 artifacts.
    pipeline_path : Path
        PRODUCTION model = complete ``sklearn`` Pipeline (TF-IDF + Linear SVM).
    faq_nlp_ready_csv : Path
        NLP-ready FAQ dataset (Module 2 output) used for answer retrieval.
    faq_dataset_csv : Path
        Final cleaned Module 1 dataset (canonical answers/entities).
    low_confidence_threshold : float
        Predictions below this score (combined with relevance) trigger fallback.
    medium_confidence_threshold : float
        Predictions between low and this value return answers with a caution.
    out_of_domain_relevance_threshold : float
        Max cosine-relevance below which a query is treated as out-of-domain.
    max_query_length : int
        Maximum characters accepted in a single user query.
    fallback_empty_message : str
        Message returned for empty / whitespace-only input.
    fallback_low_confidence_message : str
        Message returned for low-confidence predictions.
    fallback_out_of_domain_message : str
        Message returned for out-of-domain queries.
    fallback_no_answer_message : str
        Message returned when an intent has no FAQ answer.
    fallback_overflow_message : str
        Message returned for queries exceeding ``max_query_length``.
    medium_confidence_note : str
        Caution note appended to medium-confidence answers.
    """

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    model_dir: Path = field(default_factory=lambda: DEFAULT_MODEL_DIR)
    pipeline_path: Path = field(default_factory=lambda: DEFAULT_PIPELINE_PATH)
    classifier_path: Path = field(default_factory=lambda: DEFAULT_CLASSIFIER_PATH)
    vectorizer_path: Path = field(default_factory=lambda: DEFAULT_VECTORIZER_PATH)
    faq_nlp_ready_csv: Path = field(default_factory=lambda: DEFAULT_FAQ_NLP_READY_CSV)
    faq_dataset_csv: Path = field(default_factory=lambda: DEFAULT_FAQ_DATASET_CSV)

    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD
    medium_confidence_threshold: float = MEDIUM_CONFIDENCE_THRESHOLD
    out_of_domain_relevance_threshold: float = OUT_OF_DOMAIN_RELEVANCE_THRESHOLD
    max_query_length: int = MAX_QUERY_LENGTH

    fallback_empty_message: str = FALLBACK_EMPTY_MESSAGE
    fallback_low_confidence_message: str = FALLBACK_LOW_CONFIDENCE_MESSAGE
    fallback_out_of_domain_message: str = FALLBACK_OUT_OF_DOMAIN_MESSAGE
    fallback_no_answer_message: str = FALLBACK_NO_ANSWER_MESSAGE
    fallback_overflow_message: str = FALLBACK_OVERFLOW_MESSAGE
    medium_confidence_note: str = MEDIUM_CONFIDENCE_NOTE

    cli_banner: str = CLI_BANNER
    cli_help_text: str = CLI_HELP_TEXT

    # CLI commands recognized before reaching the classifier.
    exit_commands: List[str] = field(
        default_factory=lambda: ["exit", "quit", "q"]
    )
    clear_commands: List[str] = field(default_factory=lambda: ["clear", "cls"])
    help_commands: List[str] = field(default_factory=lambda: ["help", "h"])

    def validate(self) -> List[str]:
        """Return a list of configuration problems (empty == OK)."""
        problems: List[str] = []
        if self.low_confidence_threshold >= self.medium_confidence_threshold:
            problems.append(
                "low_confidence_threshold must be < medium_confidence_threshold"
            )
        if not (0.0 <= self.low_confidence_threshold <= 1.0):
            problems.append("low_confidence_threshold must be within [0, 1]")
        if not (0.0 <= self.medium_confidence_threshold <= 1.0):
            problems.append("medium_confidence_threshold must be within [0, 1]")
        if not (
            0.0 <= self.out_of_domain_relevance_threshold <= 1.0
        ):
            problems.append(
                "out_of_domain_relevance_threshold must be within [0, 1]"
            )
        if not self.pipeline_path.exists() and not (
            self.classifier_path.exists() and self.vectorizer_path.exists()
        ):
            problems.append(
                f"No model artifact found at {self.pipeline_path} or "
                f"({self.classifier_path} + {self.vectorizer_path})"
            )
        if not self.faq_nlp_ready_csv.exists() and not self.faq_dataset_csv.exists():
            problems.append("No FAQ dataset found for answer retrieval.")
        return problems

# ---------------------------------------------------------------------------
# Default configuration factory
# ---------------------------------------------------------------------------
def default_config() -> "ChatbotConfig":
    """Return a ChatbotConfig with default values derived from project root."""
    return ChatbotConfig()

