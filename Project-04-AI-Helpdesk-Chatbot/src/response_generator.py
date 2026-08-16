"""
Module 4: Response Generator
============================

Translates an intent-classification prediction into a structured,
confidence-tiered helpdesk response.

Confidence tiers (driven by Module 3's *decision-function margin* score - never
marketed as a probability):

* ``high``   (>= ``medium_confidence_threshold``) -> the retrieved FAQ answer,
  unmodified.
* ``medium`` (between low and medium thresholds) -> the FAQ answer preceded by a
  concise caution note ("moderate confidence").
* ``low``    (< ``low_confidence_threshold`` OR out-of-domain) -> no answer;
  a fallback message is returned instead.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:  # pragma: no cover - import safety
    sys.path.insert(0, str(SRC_DIR))

from chatbot_config import ChatbotConfig, get_logger  # noqa: E402
from faq_retriever import FAQRetriever  # noqa: E402
from entity_handler import EntityHandler  # noqa: E402
from fallback_handler import FallbackHandler  # noqa: E402

@dataclass
class ChatResponse:
    """Structured output of a single turn of the helpdesk chatbot."""

    query: str
    intent: str
    confidence: float
    confidence_source: str
    confidence_level: str  # "high" | "medium" | "low"
    entities: List[str] = field(default_factory=list)
    answer: str = ""
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    relevance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary view of the response."""
        return {
            "query": self.query,
            "intent": self.intent,
            "confidence": round(self.confidence, 6),
            "confidence_source": self.confidence_source,
            "confidence_level": self.confidence_level,
            "entities": list(self.entities),
            "answer": self.answer,
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
            "relevance": round(self.relevance, 6),
        }

    def __str__(self) -> str:
        return (
            f"Intent: {self.intent} | "
            f"Confidence: {self.confidence:.4f} ({self.confidence_level}) | "
            f"Answer: {self.answer}"
        )

class ResponseGenerator:
    """Generates the final answer string from a Module 3 prediction."""

    def __init__(
        self,
        faq_retriever: FAQRetriever,
        entity_handler: EntityHandler,
        fallback_handler: FallbackHandler,
        config: Optional[ChatbotConfig] = None,
    ) -> None:
        self.config = config or faq_retriever.config
        self.faq = faq_retriever
        self.entities = entity_handler
        self.fallback = fallback_handler
        self._log = get_logger()

    # ------------------------------------------------------------------ API
    def generate(
        self,
        prediction: Dict[str, Any],
        query: str,
    ) -> ChatResponse:
        """Build a :class:`ChatResponse` from a classifier prediction.

        Args:
            prediction: Output of ``IntentClassifier.predict_with_confidence``.
            query: The original user query (used for answer similarity).

        Returns:
            A fully resolved :class:`ChatResponse` (never ``None``).
        """
        intent = str(prediction["intent"])
        confidence = float(prediction["confidence"])
        confidence_source = str(prediction.get("confidence_source", "unknown"))
        detected_entities = self.entities.extract(query)
        relevance = self.faq.query_relevance(query)

        cfg = self.config
        is_out_of_domain = relevance < cfg.out_of_domain_relevance_threshold
        is_low_confidence = confidence < cfg.low_confidence_threshold

        # ---- LOW: out-of-domain or near-random confidence -> fallback
        if is_out_of_domain or is_low_confidence:
            reason = "out_of_domain" if is_out_of_domain else "low_confidence"
            return ChatResponse(
                query=query,
                intent=intent,
                confidence=confidence,
                confidence_source=confidence_source,
                confidence_level="low",
                entities=detected_entities,
                answer=self.fallback.handle_out_of_domain(),
                is_fallback=True,
                fallback_reason=reason,
                relevance=relevance,
            )

        # ---- MEDIUM: uncertain but helpdesk -> answer with caution note
        if confidence < cfg.medium_confidence_threshold:
            answer = self.faq.get_answer(intent, query)
            if answer is None:
                return self._no_answer_response(
                    intent, confidence, confidence_source,
                    detected_entities, query, relevance,
                )
            cautious = f"{cfg.medium_confidence_note}\n\n{answer}"
            self._log.info(
                "Medium-confidence response intent=%s conf=%.4f", intent,
                confidence,
            )
            return ChatResponse(
                query=query,
                intent=intent,
                confidence=confidence,
                confidence_source=confidence_source,
                confidence_level="medium",
                entities=detected_entities,
                answer=cautious,
                relevance=relevance,
            )

        # ---- HIGH: confident intent -> canonical answer
        answer = self.faq.get_answer(intent, query)
        if answer is None:
            return self._no_answer_response(
                intent, confidence, confidence_source,
                detected_entities, query, relevance,
            )
        self._log.info(
            "High-confidence response intent=%s conf=%.4f", intent, confidence,
        )
        return ChatResponse(
            query=query,
            intent=intent,
            confidence=confidence,
            confidence_source=confidence_source,
            confidence_level="high",
            entities=detected_entities,
            answer=answer,
            relevance=relevance,
        )

    # ------------------------------------------------------------ helpers
    def _no_answer_response(
        self,
        intent: str,
        confidence: float,
        confidence_source: str,
        entities: List[str],
        query: str,
        relevance: float,
    ) -> ChatResponse:
        """Build a fallback response when an intent has no FAQ answer."""
        return ChatResponse(
            query=query,
            intent=intent,
            confidence=confidence,
            confidence_source=confidence_source,
            confidence_level="low",
            entities=entities,
            answer=self.fallback.handle_no_answer(),
            is_fallback=True,
            fallback_reason="no_answer",
            relevance=relevance,
        )

