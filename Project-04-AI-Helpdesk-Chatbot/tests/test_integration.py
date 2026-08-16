"""
Integration tests for Module 5: end-to-end HTTP flow.

Tests the complete request -> validation -> chatbot engine -> intent
classifier -> FAQ retrieval -> response generator -> fallback handling
-> JSON response pipeline.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import get_api_config  # noqa: E402
from chatbot import HelpdeskChatbot  # noqa: E402
from intent_classifier import IntentClassifier, load_model  # noqa: E402
from faq_retriever import FAQRetriever  # noqa: E402
from entity_handler import EntityHandler  # noqa: E402
from fallback_handler import FallbackHandler  # noqa: E402
from response_generator import ResponseGenerator, ChatResponse  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def api_config():
    cfg = get_api_config()
    cfg.debug = False
    return cfg


@pytest.fixture(scope="session")
def chatbot(api_config):
    return HelpdeskChatbot(config=api_config.chatbot)


@pytest.fixture(scope="session")
def flask_client(api_config):
    from app import create_app
    problems = api_config.chatbot.validate()
    if problems:
        pytest.skip(f"Chatbot configuration invalid: {'; '.join(problems)}")
    app = create_app(api_config)
    return app.test_client()


# ---------------------------------------------------------------------------
# Integration tests: full HTTP flow
# ---------------------------------------------------------------------------
class TestIntegrationFlow:
    """End-to-end integration of the complete chatbot pipeline."""

    def test_password_reset_flow(self, flask_client, chatbot):
        resp = flask_client.post(
            "/api/chat",
            json={"message": "How do I reset my password?"},
        )
        assert resp.status_code == 200
        data = resp.get_json()

        # Verify intent prediction consistency.
        direct = chatbot.get_response("How do I reset my password?")
        assert data["intent"] == direct.intent

        # Verify FAQ answer consistency.
        faq_answer = chatbot.faq.get_answer(direct.intent, "How do I reset my password?")
        assert faq_answer is not None
        # The API response should contain the FAQ answer (or caution note).
        assert faq_answer in data["response"] or data["fallback"] is True

    def test_working_hours_flow(self, flask_client, chatbot):
        resp = flask_client.post(
            "/api/chat",
            json={"message": "What are the working hours?"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["intent"] == "working_hours"
        assert not data["fallback"]

    def test_laptop_problems_flow(self, flask_client, chatbot):
        resp = flask_client.post(
            "/api/chat",
            json={"message": "My laptop is not working."},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["intent"] == "laptop_problems"
        assert not data["fallback"]
        assert "laptop" in data["entities"]

    def test_fallback_flow_gibberish(self, flask_client, chatbot):
        resp = flask_client.post(
            "/api/chat",
            json={"message": "asdfghjkl random nonsense"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["fallback"] is True
        assert data["confidence"] < 0.52

    def test_fallback_flow_out_of_domain(self, flask_client, chatbot):
        resp = flask_client.post(
            "/api/chat",
            json={"message": "How many atoms are in a mountain?"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["fallback"] is True

    def test_empty_input_flow(self, flask_client, chatbot):
        resp = flask_client.post(
            "/api/chat",
            json={"message": ""},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_whitespace_input_flow(self, flask_client, chatbot):
        resp = flask_client.post(
            "/api/chat",
            json={"message": "     "},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_long_input_flow(self, flask_client, chatbot):
        resp = flask_client.post(
            "/api/chat",
            json={"message": "a" * 5001},
        )
        assert resp.status_code == 413
        data = resp.get_json()
        assert data["success"] is False

    def test_invalid_json_flow(self, flask_client, chatbot):
        resp = flask_client.post(
            "/api/chat",
            data="not json",
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_multiple_real_questions(self, flask_client, chatbot):
        questions = [
            ("How do I reset my password?", "password_reset"),
            ("My laptop is not working.", "laptop_problems"),
            ("What are the working hours?", "working_hours"),
            ("I cannot log into my email.", "email_problems"),
            ("How do I connect to WiFi?", "wifi_problems"),
            ("What is my employee ID?", "employee_id"),
            ("How do I apply for leave?", "leave_policy"),
        ]
        for question, expected_intent in questions:
            resp = flask_client.post(
                "/api/chat",
                json={"message": question},
            )
            assert resp.status_code == 200, f"Failed for: {question}"
            data = resp.get_json()
            assert data["intent"] == expected_intent, (
                f"Intent mismatch for {question}: got {data['intent']}, expected {expected_intent}"
            )
            assert "confidence" in data
            assert "response" in data
            assert "fallback" in data


# ---------------------------------------------------------------------------
# Pipeline consistency: HTTP -> classifier -> FAQ -> response -> fallback
# ---------------------------------------------------------------------------
class TestPipelineConsistency:
    """Verify that HTTP responses are consistent with direct engine calls."""

    def test_intent_consistency(self, flask_client, chatbot):
        question = "How do I reset my password?"
        direct = chatbot.get_response(question)
        resp = flask_client.post("/api/chat", json={"message": question})
        data = resp.get_json()
        assert data["intent"] == direct.intent

    def test_confidence_consistency(self, flask_client, chatbot):
        question = "How do I reset my password?"
        direct = chatbot.get_response(question)
        resp = flask_client.post("/api/chat", json={"message": question})
        data = resp.get_json()
        assert abs(data["confidence"] - direct.confidence) < 1e-4

    def test_fallback_consistency(self, flask_client, chatbot):
        question = "asdfghjkl random text"
        direct = chatbot.get_response(question)
        resp = flask_client.post("/api/chat", json={"message": question})
        data = resp.get_json()
        assert data["fallback"] is direct.is_fallback

    def test_response_text_consistency(self, flask_client, chatbot):
        question = "How do I reset my password?"
        direct = chatbot.get_response(question)
        resp = flask_client.post("/api/chat", json={"message": question})
        data = resp.get_json()
        # High confidence -> answer should contain the FAQ answer.
        assert data["response"] == direct.answer or data["fallback"] is True


# ---------------------------------------------------------------------------
# Model loads once
# ---------------------------------------------------------------------------
class TestModelSingleton:
    def test_model_not_reloaded_per_request(self, flask_client):
        import app as app_module

        initial = getattr(app_module.create_app(get_api_config()), "config", None)
        # Just verify that health endpoint is fast (< 5s), indicating cached model.
        start = time.perf_counter()
        resp = flask_client.get("/health")
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        assert elapsed < 5.0, f"Health check took {elapsed:.2f}s, model may be reloading per request"
