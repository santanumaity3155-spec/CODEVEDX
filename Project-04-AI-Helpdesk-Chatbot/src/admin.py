"""
Module 6: Admin FAQ Management
================================

Provides CRUD operations for the helpdesk FAQ dataset with validation,
safe persistence, intent validation, duplicate prevention, and logging.

Design
------
- Operates on ``data/processed/faq_nlp_ready.csv`` (the canonical FAQ file
  consumed by FAQRetriever).
- Uses atomic file writes (temp file + replace) to avoid corruption.
- Validates all input before writing.
- Validates intents against the existing ``CANONICAL_INTENTS`` list.
- Logs all admin operations at appropriate levels.
- Supports a refresh callback so the chatbot can reload updated FAQ data
  without restarting the application.
- Admin API access is gated by an optional ``ADMIN_API_KEY`` environment
  variable; if unset, access is allowed but a security warning is logged.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from nlp_preprocessor import CANONICAL_INTENTS, NLPPreprocessor
from chatbot_config import ChatbotConfig, get_logger

_log = get_logger()

# Columns required in the FAQ dataset for admin operations.
REQUIRED_COLUMNS = [
    "id", "question", "clean_question", "intent", "answer", "entity",
]

# Maximum length for FAQ question and answer fields.
MAX_QUESTION_LENGTH = 1000
MAX_ANSWER_LENGTH = 5000


class AdminAuthError(Exception):
    """Raised when admin authentication fails."""


class ValidationError(Exception):
    """Raised when FAQ input validation fails."""


class PersistenceError(Exception):
    """Raised when FAQ persistence fails."""


class FAQManager:
    """In-process FAQ CRUD manager backed by the canonical CSV dataset.

    All mutations go through :meth:`_safe_write` which writes to a temporary
    file first, validates the result, then atomically replaces the original.

    Attributes:
        faq_csv: Path to the canonical FAQ CSV file.
        config: Chatbot configuration.
        _df: In-memory DataFrame copy of the FAQ data.
        _refresh_callback: Optional callable invoked after each mutation.
    """

    def __init__(
        self,
        faq_csv: Optional[Path] = None,
        config: Optional[ChatbotConfig] = None,
        refresh_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.config = config or ChatbotConfig()
        self.faq_csv = Path(faq_csv or self.config.faq_nlp_ready_csv)
        self._refresh_callback = refresh_callback
        self._df: pd.DataFrame = self._load()
        self._intent_index: Dict[str, List[int]] = defaultdict(list)
        self._rebuild_index()
        self._next_id: int = int(self._df["id"].max()) + 1 if len(self._df) > 0 else 1
        _log.info("FAQManager initialised with %d records", len(self._df))

    # --------------------------------------------------------------- loading
    def _load(self) -> pd.DataFrame:
        """Load and validate the FAQ dataset."""
        if not self.faq_csv.exists():
            raise FileNotFoundError(
                f"FAQ dataset not found: {self.faq_csv}. "
                "Run the data preparation pipeline first."
            )
        df = pd.read_csv(self.faq_csv, encoding="utf-8")
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"FAQ dataset missing required columns: {missing}")
        df = df.dropna(subset=["intent", "answer"]).reset_index(drop=True)
        return df

    def _rebuild_index(self) -> None:
        """Rebuild the intent -> row-indices mapping."""
        self._intent_index = defaultdict(list)
        for idx, intent in enumerate(self._df["intent"].astype(str)):
            self._intent_index[str(intent)].append(idx)

    def _safe_write(self, df: pd.DataFrame) -> None:
        """Write DataFrame to the FAQ CSV safely via temp file + replace."""
        required = REQUIRED_COLUMNS
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise PersistenceError(f"Cannot write FAQ dataset: missing columns {missing}")

        # Validate critical fields before writing.
        empty_questions = df["question"].isna() | (df["question"].astype(str).str.strip() == "")
        empty_answers = df["answer"].isna() | (df["answer"].astype(str).str.strip() == "")
        if empty_questions.any():
            raise PersistenceError("Cannot write FAQ dataset: empty questions present")
        if empty_answers.any():
            raise PersistenceError("Cannot write FAQ dataset: empty answers present")

        # Write to a temp file in the same directory (ensures atomic replace works).
        parent = self.faq_csv.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".csv", prefix=".faq_tmp_")
        os.close(fd)
        try:
            df.to_csv(tmp_path, index=False, encoding="utf-8")
            # Verify the temp file is readable.
            verify = pd.read_csv(tmp_path, encoding="utf-8")
            if len(verify) != len(df):
                raise PersistenceError("Temp file verification failed: row count mismatch")
            os.replace(tmp_path, str(self.faq_csv))
            _log.info("FAQ dataset persisted: %d records", len(df))
        except Exception:
            # Clean up temp file on failure.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _notify_refresh(self) -> None:
        """Invoke the refresh callback if registered."""
        if self._refresh_callback is not None:
            try:
                self._refresh_callback()
                _log.debug("FAQ refresh callback executed successfully")
            except Exception as exc:
                _log.error("FAQ refresh callback failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------- validation
    def _validate_intent(self, intent: Any) -> str:
        """Normalize and validate an intent label.

        Returns the normalized intent string.

        Raises:
            ValidationError: if the intent is empty or not in CANONICAL_INTENTS.
        """
        if intent is None:
            raise ValidationError("Intent is required.")
        normalized = str(intent).strip().lower()
        if not normalized:
            raise ValidationError("Intent cannot be empty.")
        if normalized not in CANONICAL_INTENTS:
            raise ValidationError(
                f"Invalid intent: {normalized!r}. "
                "The supplied intent is not supported."
            )
        return normalized

    def _validate_question(self, question: Any) -> str:
        """Validate and return a question string.

        Raises:
            ValidationError: if the question is missing, empty, or too long.
        """
        if question is None:
            raise ValidationError("Question is required.")
        q = str(question).strip()
        if not q:
            raise ValidationError("Question cannot be empty.")
        if len(q) > MAX_QUESTION_LENGTH:
            raise ValidationError(
                f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters."
            )
        return q

    def _validate_answer(self, answer: Any) -> str:
        """Validate and return an answer string.

        Raises:
            ValidationError: if the answer is missing, empty, or too long.
        """
        if answer is None:
            raise ValidationError("Answer is required.")
        a = str(answer).strip()
        if not a:
            raise ValidationError("Answer cannot be empty.")
        if len(a) > MAX_ANSWER_LENGTH:
            raise ValidationError(
                f"Answer exceeds maximum length of {MAX_ANSWER_LENGTH} characters."
            )
        return a

    def _validate_entity(self, entity: Any) -> str:
        """Validate and normalize an entity value.

        Empty / None values are allowed (meaning no entity).
        Non-empty values are normalized via NLPPreprocessor.normalize_entity.
        """
        if entity is None:
            return ""
        e = str(entity).strip()
        if not e:
            return ""
        preprocessor = NLPPreprocessor(self.config.project_root)
        normalized = preprocessor.normalize_entity(e)
        return normalized

    def _check_duplicate_question(self, question: str, exclude_id: Optional[int] = None) -> bool:
        """Return True if ``question`` already exists (excluding ``exclude_id``)."""
        q = question.strip().lower()
        for _, row in self._df.iterrows():
            if exclude_id is not None and int(row["id"]) == exclude_id:
                continue
            if str(row["question"]).strip().lower() == q:
                return True
        return False

    def validate_faq_record(
        self,
        question: Any,
        intent: Any,
        answer: Any,
        entity: Any = None,
    ) -> Dict[str, str]:
        """Validate an FAQ record and return normalized fields.

        Returns:
            Dict with keys: question, intent, answer, entity, clean_question,
            tokens, filtered_tokens, lemmatized_tokens, entities.

        Raises:
            ValidationError: if any field is invalid.
        """
        q = self._validate_question(question)
        intent_val = self._validate_intent(intent)
        a = self._validate_answer(answer)
        entity_val = self._validate_entity(entity)

        preprocessor = NLPPreprocessor(self.config.project_root)
        clean_q = preprocessor.clean_text(q)
        tokens = preprocessor.tokenize_text(clean_q)
        filtered = preprocessor.remove_stopwords(tokens)
        lemmatized = preprocessor.lemmatize_tokens(filtered)
        extracted = preprocessor.extract_entities(clean_q, existing_entity=entity_val)
        primary_entity = entity_val or (extracted[0] if extracted else "")

        if not clean_q:
            raise ValidationError("Question produces empty clean_question after preprocessing.")

        return {
            "id": -1,
            "question": q,
            "clean_question": clean_q,
            "tokens": " ".join(tokens),
            "filtered_tokens": " ".join(filtered),
            "lemmatized_tokens": " ".join(lemmatized),
            "intent": intent_val,
            "answer": a,
            "entity": primary_entity,
            "entities": ",".join(extracted),
        }

    # ------------------------------------------------------------- CRUD
    def list_faqs(
        self,
        intent: Optional[str] = None,
        search: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List FAQ records with optional filtering.

        Args:
            intent: Filter by intent label (normalized).
            search: Substring search in question text (case-insensitive).
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            List of FAQ record dictionaries.
        """
        df = self._df.copy()

        if intent is not None:
            intent = str(intent).strip().lower()
            df = df[df["intent"].astype(str).str.lower() == intent]

        if search is not None:
            s = str(search).strip().lower()
            if s:
                df = df[df["question"].astype(str).str.lower().str.contains(s, na=False)]

        df = df.iloc[offset:]
        if limit is not None:
            df = df.iloc[:limit]

        records = []
        for _, row in df.iterrows():
            records.append({
                "id": int(row["id"]),
                "question": str(row["question"]),
                "intent": str(row["intent"]),
                "answer": str(row["answer"]),
                "entity": str(row.get("entity", "") or ""),
            })
        return records

    def get_faq(self, faq_id: int) -> Optional[Dict[str, Any]]:
        """Return a single FAQ record by ID, or None if not found."""
        match = self._df[self._df["id"].astype(int) == int(faq_id)]
        if len(match) == 0:
            return None
        row = match.iloc[0]
        return {
            "id": int(row["id"]),
            "question": str(row["question"]),
            "intent": str(row["intent"]),
            "answer": str(row["answer"]),
            "entity": str(row.get("entity", "") or ""),
        }

    def create_faq(
        self,
        question: Any,
        intent: Any,
        answer: Any,
        entity: Any = None,
    ) -> Dict[str, Any]:
        """Create a new FAQ record.

        Returns:
            The created FAQ record.

        Raises:
            ValidationError: if input is invalid.
            PersistenceError: if the write fails.
        """
        try:
            validated = self.validate_faq_record(question, intent, answer, entity)
        except ValidationError as exc:
            _log.warning("FAQ creation validation failed: %s", exc)
            raise

        if self._check_duplicate_question(validated["question"]):
            raise ValidationError(
                "An FAQ with this question already exists. "
                "Use update to modify an existing record."
            )

        validated["id"] = int(self._next_id)
        self._next_id += 1

        new_row = pd.DataFrame([validated])
        self._df = pd.concat([self._df, new_row], ignore_index=True)
        self._rebuild_index()
        self._safe_write(self._df)
        self._notify_refresh()

        _log.info(
            "FAQ created: id=%d intent=%s question=%.50s...",
            validated["id"], validated["intent"], validated["question"],
        )
        return {
            "id": validated["id"],
            "question": validated["question"],
            "intent": validated["intent"],
            "answer": validated["answer"],
            "entity": validated["entity"],
        }

    def update_faq(
        self,
        faq_id: int,
        question: Any = None,
        intent: Any = None,
        answer: Any = None,
        entity: Any = None,
    ) -> Dict[str, Any]:
        """Update an existing FAQ record.

        Only provided fields are updated; others remain unchanged.

        Returns:
            The updated FAQ record.

        Raises:
            ValidationError: if the ID is not found or input is invalid.
            PersistenceError: if the write fails.
        """
        match = self._df[self._df["id"].astype(int) == int(faq_id)]
        if len(match) == 0:
            raise ValidationError(f"No FAQ exists with ID {faq_id}.")

        idx = match.index[0]
        row = self._df.loc[idx]

        new_question = question if question is not None else row["question"]
        new_intent = intent if intent is not None else row["intent"]
        new_answer = answer if answer is not None else row["answer"]
        new_entity = entity if entity is not None else row.get("entity", "")

        try:
            validated = self.validate_faq_record(new_question, new_intent, new_answer, new_entity)
        except ValidationError as exc:
            _log.warning("FAQ update validation failed for id=%d: %s", faq_id, exc)
            raise
        validated["id"] = int(faq_id)

        if self._check_duplicate_question(validated["question"], exclude_id=int(faq_id)):
            raise ValidationError(
                "Another FAQ already has this question. "
                "Questions must be unique."
            )

        # Preserve NLP fields by recomputing from the validated question.
        preprocessor = NLPPreprocessor(self.config.project_root)
        validated["tokens"] = " ".join(preprocessor.tokenize_text(validated["clean_question"]))
        validated["filtered_tokens"] = " ".join(
            preprocessor.remove_stopwords(preprocessor.tokenize_text(validated["clean_question"]))
        )
        validated["lemmatized_tokens"] = " ".join(
            preprocessor.lemmatize_tokens(
                preprocessor.remove_stopwords(
                    preprocessor.tokenize_text(validated["clean_question"])
                )
            )
        )
        extracted = preprocessor.extract_entities(validated["clean_question"], existing_entity=validated["entity"])
        validated["entities"] = ",".join(extracted)

        for col in REQUIRED_COLUMNS:
            if col == "id":
                continue
            if col in validated:
                self._df.loc[idx, col] = validated[col]

        self._rebuild_index()
        self._safe_write(self._df)
        self._notify_refresh()

        _log.info("FAQ updated: id=%d intent=%s", faq_id, validated["intent"])
        return {
            "id": validated["id"],
            "question": validated["question"],
            "intent": validated["intent"],
            "answer": validated["answer"],
            "entity": validated["entity"],
        }

    def delete_faq(self, faq_id: int) -> bool:
        """Delete an FAQ record by ID.

        Returns:
            True if deleted, False if not found.

        Raises:
            PersistenceError: if the write fails.
        """
        match = self._df[self._df["id"].astype(int) == int(faq_id)]
        if len(match) == 0:
            return False

        intent_val = str(match.iloc[0]["intent"])
        self._df = self._df[self._df["id"].astype(int) != int(faq_id)].reset_index(drop=True)
        self._rebuild_index()
        self._safe_write(self._df)
        self._notify_refresh()

        _log.info("FAQ deleted: id=%d intent=%s", faq_id, intent_val)
        return True

    def reload(self) -> None:
        """Reload the FAQ dataset from disk (discarding in-memory changes)."""
        self._df = self._load()
        self._rebuild_index()
        if len(self._df) > 0:
            self._next_id = int(self._df["id"].max()) + 1
        else:
            self._next_id = 1
        _log.info("FAQManager reloaded from disk: %d records", len(self._df))

    def count(self) -> int:
        """Return the number of FAQ records."""
        return len(self._df)

    def supported_intents(self) -> List[str]:
        """Return the list of intents present in the current dataset."""
        return list(self._intent_index.keys())


def check_admin_auth(request) -> None:
    """Check admin API authentication from the Flask request.

    Reads ``ADMIN_API_KEY`` from the environment. If set, the request must
    include a matching ``X-Admin-API-Key`` header.

    Raises:
        AdminAuthError: if authentication fails.
    """
    expected_key = os.environ.get("ADMIN_API_KEY", "")
    if not expected_key:
        _log.warning(
            "Admin endpoint accessed without ADMIN_API_KEY configured. "
            "This is a security risk. Set ADMIN_API_KEY to protect admin routes."
        )
        return

    provided = request.headers.get("X-Admin-API-Key", "")
    if provided != expected_key:
        _log.warning("Admin authentication failed: invalid API key.")
        raise AdminAuthError("Invalid or missing admin API key.")
