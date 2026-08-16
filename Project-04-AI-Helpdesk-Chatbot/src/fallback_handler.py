"""
Module 4: Fallback Handler
==========================

Responsible for the *messages* returned when the chatbot cannot confidently
answer a user question.  The handler itself only owns message strings; the
decision about *when* to fall back lives in :class:`ResponseGenerator` (so the
reasoning is testable and explicit).

Fallback reasons handled
------------------------
* ``empty_input``        - user sent nothing / only whitespace.
* ``overflow``           - user sent an excessively long query.
* ``out_of_domain``      - query shares no vocabulary with the helpdesk FAQ
                            (e.g. ``"asdfghjkl"``, ``"the quick brown fox"``).
* ``low_confidence``     - intent predicted but with a near-random score.
* ``no_answer``          - intent recognised but no FAQ answer available.
"""

from __future__ import annotations

from dataclasses import dataclass

from chatbot_config import ChatbotConfig, get_logger

class FallbackHandler:
    """Produces helpdesk-appropriate fallback messages."""

    def __init__(self, config: ChatbotConfig | None = None) -> None:
        self.config = config or ChatbotConfig()
        self._log = get_logger()

    def handle_empty(self) -> str:
        """Fallback for empty / whitespace-only input."""
        self._log.info("Fallback triggered: empty_input")
        return self.config.fallback_empty_message

    def handle_overflow(self) -> str:
        """Fallback for queries exceeding the maximum length."""
        self._log.info("Fallback triggered: overflow")
        return self.config.fallback_overflow_message

    def handle_out_of_domain(self) -> str:
        """Fallback for queries unrelated to the helpdesk domain."""
        self._log.info("Fallback triggered: out_of_domain")
        return self.config.fallback_out_of_domain_message

    def handle_low_confidence(self) -> str:
        """Fallback for low-confidence intent predictions."""
        self._log.info("Fallback triggered: low_confidence")
        return self.config.fallback_low_confidence_message

    def handle_no_answer(self) -> str:
        """Fallback when an intent has no associated FAQ answer."""
        self._log.info("Fallback triggered: no_answer")
        return self.config.fallback_no_answer_message

@dataclass
class FallbackResult:
    """Structured representation of a fallback decision."""

    reason: str
    message: str
    is_fallback: bool = True

