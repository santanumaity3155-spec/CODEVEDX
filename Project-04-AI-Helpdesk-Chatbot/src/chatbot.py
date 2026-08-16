"""
Module 4: Chatbot Engine (orchestrator + CLI)
============================================

This is the production entry point for the AI Helpdesk Chatbot.  It wires
together the existing, already-trained Module 3 intent classifier with the
Module 4 response / fallback / entity / FAQ-retrieval layers:

    User query
        -> input validation          (empty / whitespace / overflow)
        -> IntentClassifier          (Modules 2 preprocessing + Module 3 model)
        -> confidence + relevance     (Module 3 decision-margin + FAQ cosine)
        -> ResponseGenerator          (HIGH / MEDIUM / LOW tiering + FAQ answer)
        -> ChatResponse               (structured, reusable)

No new model is trained.  The serialized ``models/intent_classifier_pipeline.pkl``
and the already-fitted TF-IDF vectorizer (extracted from that pipeline) are
reused directly - the vectorizer is *never* re-fit on user input (no leakage).

CLI usage
---------
Run interactively::

    python src/chatbot.py

Or use the module-level API from Python::

    from chatbot import get_response
    resp = get_response("How do I reset my password?")
    print(resp.answer)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:  # pragma: no cover - import safety
    sys.path.insert(0, str(SRC_DIR))

from chatbot_config import ChatbotConfig, get_logger  # noqa: E402
from entity_handler import EntityHandler  # noqa: E402
from fallback_handler import FallbackHandler  # noqa: E402
from faq_retriever import FAQRetriever  # noqa: E402
from response_generator import ChatResponse, ResponseGenerator  # noqa: E402
from intent_classifier import IntentClassifier, load_model  # noqa: E402

_log = get_logger()

class ChatbotLoadError(RuntimeError):
    """Raised when the chatbot cannot be initialised (missing model/data)."""

def _extract_vectorizer(classifier: IntentClassifier):
    """Reuse the *exact* fitted TF-IDF from the production pipeline.

    Extraction (never refitting) guarantees inference consistency with Module 3
    training and avoids any data leakage from user queries.
    """
    pipeline = classifier.pipeline
    named = getattr(pipeline, "named_steps", None)
    if named and "tfidf" in named:
        return named["tfidf"]
    # Component-model fallback path used by IntentClassifier.load().
    import joblib

    return joblib.load(classifier.preprocessor.project_root / "models" / "tfidf_vectorizer.pkl")

class HelpdeskChatbot:
    """Stateful-free orchestrator for a single helpdesk question turn.

    The chatbot is intentionally stateless across turns (no dialogue memory),
    which matches the single-label intent classification model it wraps.  Each
    call to :meth:`get_response` is independent and reproducible.
    """

    def __init__(
        self,
        config: Optional[ChatbotConfig] = None,
        classifier: Optional[IntentClassifier] = None,
    ) -> None:
        self.config = config or ChatbotConfig()

        problems = self.config.validate()
        if problems:
            raise ChatbotLoadError(
                "Chatbot configuration is invalid: " + "; ".join(problems)
            )

        # Module 3 intent classifier (reused, not retrained).
        self.classifier = classifier or load_model(use_cache=True)

        vectorizer = _extract_vectorizer(self.classifier)
        self.faq = FAQRetriever(
            faq_csv=self.config.faq_nlp_ready_csv,
            vectorizer=vectorizer,
            preprocessor=self.classifier.preprocessor,
            config=self.config,
        )
        self.entity_handler = EntityHandler(self.classifier.preprocessor, self.config)
        self.fallback = FallbackHandler(self.config)
        self.generator = ResponseGenerator(
            self.faq, self.entity_handler, self.fallback, self.config
        )
        _log.debug("HelpdeskChatbot initialised with %d FAQ answers", len(self.faq.df))

    def refresh_faq(self) -> None:
        """Reload the FAQ retriever from disk without restarting the model."""
        self.faq.reload()
        _log.info("Chatbot FAQ refreshed: %d records now available", len(self.faq.df))

    # ------------------------------------------------------------ validation
    def validate_input(self, query: object) -> tuple[bool, str]:
        """Validate raw user input.

        Returns ``(is_valid, payload)`` where ``payload`` is the normalized
        query string when valid, or the appropriate fallback *message* when
        invalid (empty / whitespace / overflow).
        """
        if query is None:
            return False, self.config.fallback_empty_message
        if not isinstance(query, str):
            query = str(query)
        stripped = query.strip()
        if not stripped:
            return False, self.config.fallback_empty_message
        if len(stripped) > self.config.max_query_length:
            return False, self.config.fallback_overflow_message
        return True, stripped

    # ------------------------------------------------------ classification
    def classify(self, query: str) -> Dict[str, Any]:
        """Run intent classification with confidence (Module 3 model)."""
        return self.classifier.predict_with_confidence(query)

    # ----------------------------------------------------------- response
    def get_response(self, query: object) -> ChatResponse:
        """Produce a structured :class:`ChatResponse` for a user query."""
        raw_query = "" if query is None else str(query)

        is_valid, payload = self.validate_input(query)
        if not is_valid:
            if query is None or (isinstance(query, str) and not query.strip()):
                reason = "empty_input"
            else:
                reason = "overflow"
            return ChatResponse(
                query=raw_query,
                intent="",
                confidence=0.0,
                confidence_source="invalid_input",
                confidence_level="low",
                entities=[],
                answer=payload,
                is_fallback=True,
                fallback_reason=reason,
                relevance=0.0,
            )

        prediction = self.classify(payload)
        return self.generator.generate(prediction, payload)

# --------------------------------------------------------------- rendering
def format_response(
    response: ChatResponse, config: Optional[ChatbotConfig] = None
) -> str:
    """Render a :class:`ChatResponse` as the human-facing CLI block."""
    cfg = config or ChatbotConfig()
    lines: List[str] = []
    lines.append(f"User: {response.query}")
    if response.intent:
        lines.append(f"Intent: {response.intent}")
    else:
        lines.append("Intent: (invalid input - not classified)")
    lines.append(f"Confidence: {response.confidence:.6f}")
    lines.append(f"Confidence level: {response.confidence_level}")
    lines.append(f"Source: {response.confidence_source}")
    if response.entities:
        lines.append(f"Entities: {', '.join(response.entities)}")
    lines.append(f"Relevance: {response.relevance:.4f}")
    if response.is_fallback and response.fallback_reason:
        lines.append(f"Fallback reason: {response.fallback_reason}")
    lines.append("")
    lines.append("Assistant:")
    lines.append(response.answer)
    lines.append("-" * 50)
    return "\n".join(lines)

# ----------------------------------------------------- module-level API
_default_chatbot: Optional[HelpdeskChatbot] = None

def create_chatbot(config: Optional[ChatbotConfig] = None) -> HelpdeskChatbot:
    """Construct the Module 4 helpdesk chatbot (loads the Module 3 model)."""
    return HelpdeskChatbot(config=config)

def get_response(
    query: object, chatbot: Optional[HelpdeskChatbot] = None
) -> ChatResponse:
    """Convenience: ask the chatbot a single question.

    A module-level singleton is cached so repeated calls reuse the loaded model.
    """
    global _default_chatbot
    if chatbot is None:
        if _default_chatbot is None:
            _default_chatbot = create_chatbot()
        chatbot = _default_chatbot
    return chatbot.get_response(query)

# ------------------------------------------------------------- CLI
def _clear_screen() -> None:
    """Clear the terminal screen in a cross-platform way."""
    os.system("cls" if os.name == "nt" else "clear")

def _classify_command(text: str, config: ChatbotConfig) -> Optional[str]:
    """Return the command kind if ``text`` is a CLI command, else None."""
    lowered = text.strip().lower()
    if lowered in config.exit_commands:
        return "exit"
    if lowered in config.clear_commands:
        return "clear"
    if lowered in config.help_commands:
        return "help"
    return None

def chat_loop(config: Optional[ChatbotConfig] = None) -> int:
    """Run the interactive CLI until the user exits."""
    cfg = config or ChatbotConfig()
    bot = create_chatbot(cfg)
    print(cfg.cli_banner)
    print("Type a question, or type 'help' for commands.")
    while True:
        try:
            raw = input("\nUser: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0
        if raw is None:
            continue
        command = _classify_command(raw, cfg)
        if command == "exit":
            print("Goodbye!")
            return 0
        if command == "clear":
            _clear_screen()
            print(cfg.cli_banner)
            continue
        if command == "help":
            print(cfg.cli_help_text)
            continue
        response = bot.get_response(raw)
        print()
        print(format_response(response, cfg))
    return 0

def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: interactive chat or single-question mode."""
    parser = argparse.ArgumentParser(
        prog="python src/chatbot.py",
        description="AI Helpdesk Chatbot (Module 4) - intent classification + FAQ.",
    )
    parser.add_argument(
        "--once", "-q", metavar="QUESTION",
        help="Answer a single question non-interactively and exit.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="With --once, print the ChatResponse as JSON.",
    )
    args = parser.parse_args(argv)

    if args.once is not None:
        response = get_response(args.once)
        if args.json:
            import json

            print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
        else:
            cfg = ChatbotConfig()
            print(format_response(response, cfg))
        return 0

    return chat_loop()

if __name__ == "__main__":
    sys.exit(main())

