"""
Test suite for Module 4: AI Helpdesk Chatbot engine (real behaviour).

These tests exercise the *actual* chatbot behaviour end-to-end.  They cover:
module imports, model loading, valid prediction, known intents, known FAQ
responses, confidence handling, high-confidence response, low-confidence
fallback, empty / whitespace / malformed input, unknown questions, response
generation, FAQ retrieval, entity handling, sequential questions, inference
consistency, model reload and regression protection for Modules 1-3.
No "file exists"-only assertions are used.
"""

import sys
from pathlib import Path

import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlp_preprocessor import CANONICAL_INTENTS  # noqa: E402
from intent_classifier import IntentClassifier, load_model  # noqa: E402
from chatbot_config import ChatbotConfig  # noqa: E402
from chatbot import (  # noqa: E402
    HelpdeskChatbot, create_chatbot, get_response, format_response,
    ChatResponse, ChatbotConfig as _CfgAlias,
)
from faq_retriever import FAQRetriever  # noqa: E402
from entity_handler import EntityHandler  # noqa: E402
from fallback_handler import FallbackHandler  # noqa: E402
from response_generator import ResponseGenerator  # noqa: E402

FAQ_CSV = PROJECT_ROOT / "data" / "processed" / "faq_nlp_ready.csv"

# Verified (question -> expected intent) pairs, calibrated against the
# deterministic Module 3 Linear SVM.
KNOWN_INTENTS = {
    "How do I reset my password?": "password_reset",
    "How do I change my password?": "password_reset",
    "What are the office timings?": "working_hours",
    "What are the working hours?": "working_hours",
    "When will I receive my salary?": "salary_information",
    "My laptop is not working.": "laptop_problems",
    "I cannot log into my email.": "email_problems",
    "How do I apply for leave?": "leave_policy",
    "I forgot my company password.": "password_reset",
    "What is my employee ID?": "employee_id",
    "How do I connect to WiFi?": "wifi_problems",
}

@pytest.fixture(scope="session")
def bot() -> HelpdeskChatbot:
    """Real chatbot instance (loads the Module 3 model once)."""
    return create_chatbot()

@pytest.fixture(scope="session")
def faq(bot: HelpdeskChatbot) -> FAQRetriever:
    return bot.faq

# 1. Module imports ----------------------------------------------------------
def test_module_imports():
    assert HelpdeskChatbot is not None
    assert create_chatbot is not None
    assert get_response is not None
    assert format_response is not None
    assert ChatResponse is not None
    assert ChatbotConfig is not None
    assert FAQRetriever is not None
    assert EntityHandler is not None
    assert FallbackHandler is not None
    assert ResponseGenerator is not None
    for attr in ("get_response", "validate_input", "get_response"):
        assert hasattr(HelpdeskChatbot, attr)

# 2. Model loading -----------------------------------------------------------
def test_model_loading(bot):
    assert bot.classifier is not None
    assert len(bot.classifier.classes) == 22
    assert set(bot.classifier.classes) >= {"password_reset", "wifi_problems"}

# 3. Valid prediction --------------------------------------------------------
def test_valid_prediction(bot):
    resp = bot.get_response("How do I reset my password?")
    assert isinstance(resp, ChatResponse)
    assert resp.intent == "password_reset"
    assert resp.confidence > 0
    assert resp.answer  # non-empty answer
    assert not resp.is_fallback

# 4. Known intents -----------------------------------------------------------
def test_known_intents(bot):
    for question, expected_intent in KNOWN_INTENTS.items():
        resp = bot.get_response(question)
        assert resp.intent == expected_intent, (
            f"{question!r} -> got {resp.intent}, expected {expected_intent}"
        )

# 5. Known FAQ responses ------------------------------------------------------
def test_known_faq_responses(bot):
    for question, expected_intent in KNOWN_INTENTS.items():
        resp = bot.get_response(question)
        valid_answers = set(bot.faq.get_intent_answers(expected_intent))
        assert valid_answers, f"no FAQ answers stored for intent {expected_intent}"
        # The high/medium tier returns a real FAQ answer (or a caution+answer).
        # Strip the caution note prefix for the assertion.
        answer = resp.answer
        if answer.startswith("[Note:"):
            note_end = answer.index("]") + 1
            answer = answer[note_end:].strip()
        assert answer in valid_answers or answer == bot.faq.get_answer(
            expected_intent, question
        ), (
            f"answer for {question!r} not found among FAQ answers for "
            f"{expected_intent}"
        )

# 6. Confidence handling ----------------------------------------------------
def test_confidence_handling(bot):
    resp = bot.get_response("How do I reset my password?")
    # Linear SVM -> decision-function margin confidence, NOT a probability.
    assert resp.confidence_source == "decision_function_margin"
    assert 0.0 < resp.confidence <= 1.0
    # Must not advertise probabilities for a non-probabilistic model.
    assert "predict_proba" not in resp.confidence_source

# 7. High-confidence response ----------------------------------------------
def test_high_confidence_response(bot):
    resp = bot.get_response("My laptop is not working.")
    assert resp.confidence_level == "high"
    assert resp.confidence >= bot.config.medium_confidence_threshold
    # High-confidence answers must be the raw FAQ answer (no caution note).
    assert not resp.answer.startswith("[Note:")
    assert not resp.is_fallback

# 8. Low-confidence fallback ------------------------------------------------
def test_low_confidence_fallback(bot):
    resp = bot.get_response("asdfghjkl random text")
    assert resp.is_fallback is True
    assert resp.confidence_level == "low"
    assert "rephrase" in resp.answer.lower()

# 9. Empty input ------------------------------------------------------------
def test_empty_input(bot):
    for empty in ("", "   ", "\t\n  "):
        resp = bot.get_response(empty)
        assert resp.is_fallback is True
        assert resp.fallback_reason == "empty_input"
        assert resp.confidence == 0.0
        assert resp.intent == ""
        assert resp.answer == bot.config.fallback_empty_message

# 10. Whitespace input -------------------------------------------------------
def test_whitespace_input(bot):
    resp = bot.get_response("      \n\t  \n")
    assert resp.is_fallback is True
    assert resp.fallback_reason == "empty_input"
    assert resp.answer == bot.config.fallback_empty_message
    assert not bot.faq.query_relevance("      \n") > 0

# 11. Malformed input --------------------------------------------------------
def test_malformed_input(bot):
    # None -> empty-input fallback (no exception).
    resp = bot.get_response(None)
    assert resp.is_fallback is True
    assert resp.fallback_reason == "empty_input"
    # Non-string scalar -> coerced to string, then handled gracefully.
    resp_num = bot.get_response(123456789)
    assert isinstance(resp_num, ChatResponse)
    assert resp_num.is_fallback is True
    assert resp_num.confidence_level == "low"

# 12. Unknown / unrelated question ------------------------------------------
def test_unknown_question(bot):
    # Genuinely helpdesk-out-of-domain text -> must fall back, not hallucinate.
    resp = bot.get_response("How many atoms are in a mountain?")
    assert resp.is_fallback is True
    assert resp.confidence_level == "low"

# 13. Response generation ----------------------------------------------------
def test_response_generation(bot):
    resp = bot.get_response("How do I reset my password?")
    d = resp.to_dict()
    for key in (
        "query", "intent", "confidence", "confidence_source",
        "confidence_level", "entities", "answer", "is_fallback",
        "fallback_reason", "relevance",
    ):
        assert key in d, f"ChatResponse.to_dict missing {key}"
    assert d["intent"] == "password_reset"
    assert d["confidence_level"] == "high"
    assert isinstance(d["entities"], list)
    assert str(resp)  # __str__ works
    # Module-level convenience API
    resp2 = get_response("How do I reset my password?", chatbot=bot)
    assert resp2.intent == resp.intent

# 14. FAQ retrieval ----------------------------------------------------------
def test_faq_retrieval(faq):
    assert FAQ_CSV.exists()
    assert "password_reset" in faq.supported_intents()
    answers = faq.get_intent_answers("password_reset")
    assert len(answers) >= 1
    assert all(a and a.strip() for a in answers)
    # Canonical entity per intent is dataset-derived.
    assert faq.get_entity("password_reset") == "password"
    # Query-aware retrieval returns an FAQ answer.
    ans = faq.get_answer("password_reset", "How do I reset my password?")
    assert ans in answers
    # Unknown intent -> None (graceful).
    assert faq.get_answer("does_not_exist") is None

# 15. Entity handling --------------------------------------------------------
def test_entity_handling(bot):
    resp = bot.get_response("My laptop is not working.")
    assert "laptop" in resp.entities
    resp2 = bot.get_response("My laptop cannot connect to Wi-Fi.")
    ents = set(resp2.entities)
    assert "laptop" in ents and "wifi" in ents
    # Question with no entity -> empty entity list (not an error).
    resp3 = bot.get_response("What time is it?")
    assert isinstance(resp3.entities, list)

# 16. Multiple sequential questions ------------------------------------------
def test_multiple_sequential_questions(bot):
    sequence = [
        ("How do I reset my password?", "password_reset"),
        ("My laptop is not working.", "laptop_problems"),
        ("When will I receive my salary?", "salary_information"),
    ]
    results = [bot.get_response(q) for q, _ in sequence]
    for (q, expected), res in zip(sequence, results):
        assert res.intent == expected, f"{q!r} -> {res.intent}"
    # Each turn independent (stateless).
    assert all(r is not None for r in results)

# 17. Inference consistency ----------------------------------------------------
def test_inference_consistency(bot):
    q = "How do I reset my password?"
    r1 = bot.get_response(q).to_dict()
    r2 = bot.get_response(q).to_dict()
    assert r1 == r2

# 18. Model reload -----------------------------------------------------------
def test_model_reload():
    m1 = load_model(use_cache=False)
    m2 = load_model(use_cache=False)
    q = "How do I reset my password?"
    p1 = m1.predict_with_confidence(q)
    p2 = m2.predict_with_confidence(q)
    assert p1["intent"] == p2["intent"]
    assert p1["confidence"] == pytest.approx(p2["confidence"])

# 19. Regression protection for Modules 1-3 ---------------------------------
def test_modules_regression(bot):
    # Module 1 & 2 data integrity: the FAQ the chatbot answers from is the
    # NLP-ready dataset (Module 2 output), itself derived from Module 1.
    assert FAQ_CSV.exists()
    df = pd.read_csv(FAQ_CSV)
    assert len(df) >= 100  # Module 1+2 produced 294 records
    assert set(df["intent"].astype(str)) <= set(CANONICAL_INTENTS)
    # Module 3 contract: the production model is a fitted pipeline exposing
    # decision_function (Linear SVM) -> bounded margin confidence.
    assert hasattr(bot.classifier.pipeline, "decision_function") or hasattr(
        bot.classifier.pipeline, "predict_proba"
    )
    # The exact training-time preprocessing is reused at inference.
    from nlp_preprocessor import NLPPreprocessor

    prep = NLPPreprocessor(PROJECT_ROOT)
    q = "How do I reset my password?"
    assert bot.classifier.preprocess_text(q) == prep.clean_text(q)

