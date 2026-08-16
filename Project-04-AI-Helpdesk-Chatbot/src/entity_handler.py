"""
Module 4: Entity Handler
========================

Wraps the existing rule-based entity extraction from Module 2
(:meth:`NLPPreprocessor.extract_entities`) and exposes only the *detection* of
entities - it never invents entities or entities outside the canonical
dataset/entity schema defined in ``src/nlp_preprocessor.py``.

The extracted entities are surfaced to the caller so that responses can be made
"entity-aware" (e.g. acknowledging "laptop" / "wifi") without fabricating
domain facts.  The factual answer text itself always comes from the FAQ
dataset (see :class:`faq_retriever.FAQRetriever`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:  # pragma: no cover - import safety
    sys.path.insert(0, str(SRC_DIR))

from chatbot_config import ChatbotConfig, get_logger  # noqa: E402
from nlp_preprocessor import NLPPreprocessor  # noqa: E402

class EntityHandler:
    """Detects canonical entities in a user query.

    Detection is fully dataset-driven: it reuses the ``ENTITY_KEYWORDS``
    controlled vocabulary and word-boundary matching already shipped in
    Module 2, so the entity schema never drifts from the training data.
    """

    def __init__(
        self,
        preprocessor: Optional[NLPPreprocessor] = None,
        config: Optional[ChatbotConfig] = None,
    ) -> None:
        self.config = config or ChatbotConfig()
        self.preprocessor = preprocessor or NLPPreprocessor(
            self.config.project_root
        )
        self._log = get_logger()

    def extract(self, text: object) -> List[str]:
        """Return canonical entities detected in ``text`` (deduplicated)."""
        if not isinstance(text, str):
            return []
        return self.preprocessor.extract_entities(text)

    def get_primary_entity(self, text: object) -> str:
        """Return the first detected entity, or ``""`` when none."""
        entities = self.extract(text)
        return entities[0] if entities else ""

    def has_entity(self, text: object, entity: str) -> bool:
        """Return True if ``entity`` (canonical) is present in ``text``."""
        return str(entity) in self.extract(text)

    def entity_summary(self, text: object) -> str:
        """Return a short human-readable mention of detected entities.

        Returns ``""`` when no entity is detected (so responses stay silent on
        entities unless one is actually relevant).
        """
        entities = self.extract(text)
        self._log.debug("Detected entities: %s", entities)
        if not entities:
            return ""
        if len(entities) == 1:
            return f" (entity: {entities[0]})"
        return f" (entities: {', '.join(entities)})"

