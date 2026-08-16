"""
Module 4: FAQ Retriever
=======================

Responsible for retrieving the most appropriate helpdesk *answer* for a
predicted intent.  It is built directly on the existing artifacts:

* ``data/processed/faq_nlp_ready.csv`` - the Module 2 NLP-ready dataset, which
  carries the canonical ``question`` / ``clean_question`` / ``intent`` /
  ``answer`` / ``entity`` schema.
* The **already-fitted** TF-IDF vectorizer inside ``models/
  intent_classifier_pipeline.pkl``.

Data-leakage policy
-------------------
The fitted vectorizer is **never re-fit** on user input.  User queries are only
*transformed* using the existing vocabulary/transform, exactly as Module 3 does
for intent classification.  This keeps inference consistent with training.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:  # pragma: no cover - import safety
    sys.path.insert(0, str(SRC_DIR))

from chatbot_config import ChatbotConfig, get_logger  # noqa: E402
from nlp_preprocessor import NLPPreprocessor  # noqa: E402

REQUIRED_COLUMNS = ["question", "clean_question", "intent", "answer"]

class FAQRetriever:
    """Retrieval layer: maps a predicted intent to a concrete helpdesk answer.

    Answers are selected data-driven (by cosine similarity of the user query to
    the FAQ questions belonging to the predicted intent) rather than hard-coded,
    so the same factual answers stored in the Module 1 dataset are reused.
    """

    def __init__(
        self,
        faq_csv: Optional[Path] = None,
        vectorizer=None,
        preprocessor: Optional[NLPPreprocessor] = None,
        config: Optional[ChatbotConfig] = None,
    ) -> None:
        self.config = config or ChatbotConfig()
        self.faq_csv = Path(faq_csv or self.config.faq_nlp_ready_csv)
        self.vectorizer = vectorizer
        self.preprocessor = preprocessor or NLPPreprocessor(self.config.project_root)
        self._log = get_logger()

        self.df: pd.DataFrame = self._load()
        self._intent_index: Dict[str, List[int]] = defaultdict(list)
        self._build_index()
        self._corpus_vectors = self._vectorize_corpus()

    # --------------------------------------------------------------- loading
    def _load(self) -> pd.DataFrame:
        """Load and validate the FAQ dataset."""
        if not self.faq_csv.exists():
            raise FileNotFoundError(
                f"FAQ dataset not found: {self.faq_csv}. "
                "Run `python src/prepare_dataset.py` and "
                "`python src/prepare_nlp_data.py` first."
            )
        df = pd.read_csv(self.faq_csv, encoding="utf-8")
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"FAQ dataset missing required columns: {missing}")
        df = df.dropna(subset=["intent", "answer"]).reset_index(drop=True)
        return df

    # ----------------------------------------------------------- indexing
    def _build_index(self) -> None:
        """Build an intent -> list-of-row-indices mapping for fast lookup."""
        for idx, intent in enumerate(self.df["intent"].astype(str)):
            self._intent_index[str(intent)].append(idx)

    def _vectorize_corpus(self) -> "np.ndarray":
        """Pre-vectorize the entire corpus once (using the fitted vectorizer)."""
        if self.vectorizer is None:
            import joblib

            self.vectorizer = joblib.load(self.config.vectorizer_path)
        clean_questions = self.df["clean_question"].astype(str).tolist()
        return self.vectorizer.transform(clean_questions)

    # ------------------------------------------------------------------ API
    def supported_intents(self) -> List[str]:
        """Return the list of intents the FAQ dataset contains answers for."""
        return list(self._intent_index.keys())

    def supports_intent(self, intent: str) -> bool:
        """Return True if the FAQ has at least one answer for ``intent``."""
        return str(intent) in self._intent_index

    def get_intent_answers(self, intent: str) -> List[str]:
        """Return all FAQ answers associated with ``intent``."""
        intent = str(intent)
        if not self.supports_intent(intent):
            return []
        rows = self.df.iloc[self._intent_index[intent]]
        return [str(a) for a in rows["answer"].tolist()]

    def get_entity(self, intent: str) -> str:
        """Return the canonical dataset entity for an intent (first occurrence)."""
        intent = str(intent)
        if not self.supports_intent(intent):
            return ""
        rows = self.df.iloc[self._intent_index[intent]]
        entity = rows["entity"].dropna().astype(str)
        non_empty = [e for e in entity if e.strip()]
        return non_empty[0] if non_empty else ""

    def query_relevance(self, query: str) -> float:
        """Max cosine similarity of the cleaned query against the FAQ corpus.

        A value near 0.0 means the query shares no vocabulary with any helpdesk
        FAQ (likely out-of-domain).  This reuses the *fitted* vectorizer only.
        """
        if not isinstance(query, str) or not query.strip():
            return 0.0
        cleaned = self.preprocessor.clean_text(query)
        if not cleaned:
            return 0.0
        query_vec = self.vectorizer.transform([cleaned])
        sims = cosine_similarity(query_vec, self._corpus_vectors)[0]
        return float(np.max(sims))

    def get_answer(
        self, intent: str, query: Optional[str] = None
    ) -> Optional[str]:
        """Retrieve the best answer for ``intent`` (optionally query-aware).

        When ``query`` is provided, the answer is taken from the FAQ question
        that is most semantically similar to the user's query *within* the
        predicted intent.  Without ``query``, the canonical (first) answer for
        the intent is returned.

        Returns ``None`` when the intent has no FAQ answer.
        """
        intent = str(intent)
        if not self.supports_intent(intent):
            self._log.info("No FAQ answer for intent %r", intent)
            return None

        indices = self._intent_index[intent]
        if query is None or not isinstance(query, str) or not query.strip():
            return str(self.df.iloc[indices[0]]["answer"])

        cleaned = self.preprocessor.clean_text(query)
        if not cleaned:
            return str(self.df.iloc[indices[0]]["answer"])

        query_vec = self.vectorizer.transform([cleaned])
        intent_vectors = self._corpus_vectors[indices]
        sims = cosine_similarity(query_vec, intent_vectors)[0]
        best_local = int(np.argmax(sims))
        best_idx = indices[best_local]
        return str(self.df.iloc[best_idx]["answer"])

    def lookup_intent_for_answer(self, answer: str) -> Optional[str]:
        """Return the intent whose FAQ answer text matches ``answer``."""
        if not isinstance(answer, str) or not answer.strip():
            return None
        matches = self.df.index[self.df["answer"].astype(str) == answer].tolist()
        if not matches:
            return None
        return str(self.df.iloc[matches[0]]["intent"])

